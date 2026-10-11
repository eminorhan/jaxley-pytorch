# This file is part of jaxley-pytorch, a PyTorch port of Jaxley, a differentiable
# neuroscience simulator. Jaxley is licensed under the Apache License Version 2.0, see
# <https://www.apache.org/licenses/>

from typing import Callable, Tuple

import numpy as np
import torch
from torch import Tensor as Array


def _remove_currents_from_states(states: dict[str, Array], current_keys: list[str]):
    """Shallow copy of the states, excluding the currents through channels and synapses.

    Args:
        states: States (including currents) of the system.
        current_keys: The names of all channel currents.
    """
    return {key: value for key, value in states.items() if key not in current_keys}


def build_dynamic_state_utils(module) -> Tuple[Callable, Callable, Callable, Callable]:
    r"""Return functions which extract the dynamic (ODE) states of a ``jx.Module``.

    These utility functions are meant to be used together with
    ``jx.integrate.build_init_and_step_fn``. The ``init_fn`` returned by
    ``build_init_and_step_fn`` returns an ``all_states``, which is a dictionary
    of all states, including observables: the voltages at branchpoints, the channel
    and synapse currents, and NaN elements for channel states which do not exist
    in certain compartments. The utility functions returned by
    ``build_utils_for_dynamic_states()`` modify the ``all_states`` as follows:

    - They remove all channel currents, synapse currents, and branchpoint voltages
      (which can be computed from compartment voltages). Additionally, if states
      are only defined on a subset of compartments, the NaN padding is removed.
      As such, only "true" dynamic states remain. This is handled by the returned
      functions ``remove_observables`` and ``add_observables``.
    - They return the states as a flat tensor. This is handled by the returned
      functions ``flatten`` and ``unflatten``.

    The states are flattened along their last dimension (in the order of the sorted
    state names, as `jax.flatten_util.ravel_pytree` does), so leading batch dimensions
    are preserved.

    .. warning::

        If a membrane current is used as a state by another channel (for example,
        the calcium current ``i_ca`` used in the ``channel_states`` of a
        calcium-dependent potassium channel), then this current will be included
        in the returned "true" ODE states. Similarly, if there are ``channel_states``
        that are directly computed based on other states (e.g., the calcium reversal
        potential via the nernst equation), they are also considered "true" ODE states.
        Please remove such states manually if desired.

    Args:
        module: A ``Module`` object, e.g., a ``jx.Cell``.

    Returns:

        * ``remove_observables(all_states)``: removes the membrane currents, synaptic
          currents, branchpoint voltages and NaN padding from the states dict.
        * ``add_observables(dynamic_states_pytree, all_params, delta_t)``: adds
          membrane currents, synaptic currents, and branchpoint voltages to the states
          dictionary.
        * ``flatten(dynamic_states_pytree)``: flattens dynamic states into a tensor.
        * ``unflatten(flat_states)``: converts the state vector back to a dict.

    Example usage
    ^^^^^^^^^^^^^

    Build a ``step_dynamics`` function and use it to compute the Jacobian of a single
    step.

    ::

        from torch.func import jacfwd

        import jaxley_pytorch as jx
        from jaxley_pytorch.integrate import build_init_and_step_fn
        from jaxley_pytorch.utils.dynamics import build_dynamic_state_utils
        from jaxley_pytorch.channels import Leak

        comp = jx.Compartment()
        branch = jx.Branch(comp, 2)
        cell = jx.Cell(branch, parents=[-1, 0, 0])
        cell.insert(Leak())
        params = cell.get_parameters()

        externals = cell.externals.copy()
        external_inds = cell.external_inds.copy()

        init_fn, step_fn = build_init_and_step_fn(cell)
        remove_observables, add_observables, flatten, unflatten = build_dynamic_state_utils(cell)

        all_states, all_params = init_fn(params)
        dynamic_states = flatten(remove_observables(all_states))

        def step_dynamics(dynamic_states):
            all_states = add_observables(unflatten(dynamic_states), all_params, 0.025)
            all_states = step_fn(
                all_states, all_params, externals, external_inds, delta_t=0.025
            )
            return flatten(remove_observables(all_states))

        jacobian = torch.autograd.functional.jacobian(step_dynamics, dynamic_states)
    """

    all_states = module.get_all_states([])
    names_of_currents = module.membrane_current_names + module.synapse_current_names

    # Currents that are also used as membrane states (e.g., the calcium current
    # ``i_ca`` used in the ``channel_states`` of a calcium-dependent potassium channel),
    # then we treat it as a "true" state. If we would not do this, the
    # ``add_observables`` method below would be very tricky to implement (because
    # states depend on currents and currents depend on states).
    names_of_currents = [
        current
        for current in names_of_currents
        if current not in list(all_states.keys())
    ]

    # Keys corresponding to membrane states and not to synapse states.
    membrane_states_keys = set(module.torchnodes.keys())
    original_length = all_states["v"].shape[-1]
    device = all_states["v"].device

    def _is_valid_membrane_leaf(key: str, leaf) -> bool:
        return key in membrane_states_keys and isinstance(leaf, torch.Tensor) and (
            leaf.ndim > 0
        )

    # Remove branchpoints if needed.
    # ----------------------------------------------------------

    # Check whether the module defines branchpoints and whether any exist.
    if hasattr(module, "_branchpoints") and len(module._branchpoints.index) > 0:
        # Indices of compartments that correspond to branchpoints.
        filter_indices = module._branchpoints.index.to_numpy().astype(int)
        # Indices to keep (non-branchpoints).
        keep = np.setdiff1d(
            np.arange(original_length), filter_indices, assume_unique=True
        )
        # If there are branchpoints, we need to remember that we will apply the filter.
        branch_filter_applied = True
    # If there are no branchpoints, we don't need to filter.
    else:
        keep = np.arange(original_length)
        branch_filter_applied = False
    keep_indices = torch.as_tensor(keep, device=device)

    def _remove_branchpoints(states):
        return {
            k: (v[..., keep_indices] if _is_valid_membrane_leaf(k, v) else v)
            for k, v in states.items()
        }

    all_states = _remove_branchpoints(all_states)

    # Number of compartments after branchpoint removal.
    filtered_length = all_states["v"].shape[-1]

    # Remove NaNs (appear if some states are not defined on all compartments)
    # ----------------------------------------------------------
    nan_indices_tree = {
        k: torch.nonzero(~torch.isnan(v), as_tuple=True)[-1]
        for k, v in all_states.items()
        if isinstance(v, torch.Tensor) and v.ndim > 0
    }

    def _remove_nans(states):
        return {
            k: (v[..., nan_indices_tree[k]] if _is_valid_membrane_leaf(k, v) else v)
            for k, v in states.items()
        }

    all_states_no_nans = _remove_nans(all_states)

    # Flatten membrane states into a single vector.
    # ----------------------------------------------------------
    template_keys = sorted(all_states_no_nans.keys())
    template_sizes = [
        int(np.prod(all_states_no_nans[k].shape[-1:])) if all_states_no_nans[k].ndim
        else 1
        for k in template_keys
    ]

    def flatten(dynamic_states_pytree: dict[str, Array]) -> Array:
        """Flatten dynamic states (a dict of tensors) into a tensor.

        Args:
            dynamic_states_pytree: Dynamic states as dict of tensors. Contains all
                dynamic states.

        Returns:
            Flattened dynamic states as a tensor of shape `(..., num_states)`.
        """
        keys = sorted(dynamic_states_pytree.keys())
        values = [torch.as_tensor(dynamic_states_pytree[k]) for k in keys]
        values = [v[..., None] if v.ndim == 0 else v for v in values]
        batch = torch.broadcast_shapes(*[v.shape[:-1] for v in values])
        values = [torch.broadcast_to(v, (*batch, v.shape[-1])) for v in values]
        return torch.cat(values, dim=-1)

    def unflatten(flat_states: Array) -> dict[str, Array]:
        """Convert the state vector back to a dict of tensors."""
        pieces = torch.split(flat_states, template_sizes, dim=-1)
        return {k: p for k, p in zip(template_keys, pieces)}

    # Now we can create functions that convert between the full state pytree
    # and the filtered state vector.
    # ----------------------------------------------------------

    def remove_observables(states: dict[str, Array]) -> dict[str, Array]:
        r"""Remove the membrane currents, synaptic currents, and branchpoint voltages.

        Thus, the returned states only include true "dynamic" states.

         Args:
            all_states:. All observable states of the system, including membrane
                and synaptic currents, branchpoint voltages, and NaN-padded states
                in cases where some mechanisms exist only in some compartments.

        Returns:
            All dynamic states of the system.
        """
        filtered_states = _remove_currents_from_states(states, names_of_currents)

        # Removes branchpoints, only from membrane states.
        filtered_states = _remove_branchpoints(filtered_states)

        # Remove NaN-containing compartments, again only from membrane states.
        filtered_states = _remove_nans(filtered_states)
        return filtered_states

    def _restore_leaf(filtered_array, nan_indices_leaf):
        """Restore NaN padding."""
        restored = torch.full(
            (*filtered_array.shape[:-1], filtered_length),
            float("nan"),
            dtype=filtered_array.dtype,
            device=filtered_array.device,
        )
        return restored.index_copy(-1, nan_indices_leaf, filtered_array)

    def _restore_branch_leaf(leaf):
        """Restore branchpoint voltages"""
        # Branchpoint states are set to -1.0 in the module.torchnodes.
        restored = torch.full(
            (*leaf.shape[:-1], original_length),
            -1.0,
            dtype=leaf.dtype,
            device=leaf.device,
        )
        return restored.index_copy(-1, keep_indices, leaf)

    def add_observables(
        dynamic_states_pytree: dict[str, Array],
        all_params: dict[str, Array],
        delta_t: float,
    ) -> dict[str, Array]:
        """Add membrane currents, synaptic currents, and branchpoint voltages to states.

        Args:
            dynamic_states_pytree: Contains all dynamic states of the module,
                formatted as a dictionary of tensors.
            all_params: Contains _all_ parameters that are needed to simulate the
                system.
            delta_t: The time step.

        Returns:
            ``all_states`` which can be passed to the ``step_fn`` (returned by
            ``jx.integrate.build_init_and_step_fn``).
        """
        # Restore NaN padding.
        restored_states = {
            k: (
                _restore_leaf(v, nan_indices_tree[k])
                if _is_valid_membrane_leaf(k, v)
                else v
            )
            for k, v in dynamic_states_pytree.items()
        }

        # Restore branchpoint voltages if there were any branchpoints.
        if branch_filter_applied:
            restored_states = {
                k: (_restore_branch_leaf(v) if _is_valid_membrane_leaf(k, v) else v)
                for k, v in restored_states.items()
            }

        # Add channel currents to the restored states.
        restored_states = module.append_channel_currents_to_states(
            restored_states, all_params, delta_t=delta_t
        )
        return restored_states

    return remove_observables, add_observables, flatten, unflatten
