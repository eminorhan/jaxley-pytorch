# This file is part of jaxley-pytorch, a PyTorch port of Jaxley, a differentiable
# neuroscience simulator. Jaxley is licensed under the Apache License Version 2.0, see
# <https://www.apache.org/licenses/>
from math import prod
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from jaxley_pytorch.modules import Module
from jaxley_pytorch.utils.cell_utils import params_to_pstate
from jaxley_pytorch.utils.torch_utils import (
    as_index,
    first_float_tensor,
    nested_checkpoint_scan,
    tree_tensors,
)


def _resolve_voltage_solver(module: Module, voltage_solver: str) -> str:
    if voltage_solver == "jaxley.dhs":
        # Automatically infer the voltage solver.
        if module._solver_device in ["gpu", "tpu"]:
            return "jaxley.dhs.gpu"
        return "jaxley.dhs.cpu"
    return voltage_solver


def _batch_shape(tensors) -> Tuple[int, ...]:
    """Broadcast leading (batch) dimensions of tensors whose last dim is not batch."""
    shapes = [t.shape[:-1] for t in tensors if isinstance(t, Tensor) and t.ndim > 0]
    return tuple(torch.broadcast_shapes(*shapes)) if shapes else ()


def _expand_states(states: Dict[str, Tensor], batch: Tuple[int, ...]) -> Dict:
    """Give every state the full batch shape `(*batch, n)`."""
    expanded = {}
    for key, val in states.items():
        if not isinstance(val, Tensor):
            expanded[key] = val
            continue
        if val.ndim == 0:
            expanded[key] = val
            continue
        target = (*batch, val.shape[-1])
        if tuple(val.shape) != target:
            val = torch.broadcast_to(val, target).contiguous()
        expanded[key] = val
    return expanded


def _build_recorder(module: Module, device: torch.device) -> Callable:
    """Return `record(states) -> Tensor` of shape `(..., num_recordings)`.

    Recordings of the same state are gathered with a single indexing operation.
    """
    rec_inds = module.recordings.rec_index.to_numpy().astype(int)
    rec_states = module.recordings.state.to_numpy()
    groups = []
    order = []
    for name in dict.fromkeys(rec_states.tolist()):
        positions = np.nonzero(rec_states == name)[0]
        groups.append((name, as_index(rec_inds[positions], device)))
        order.append(positions)
    order = np.concatenate(order)
    num = len(rec_inds)
    if len(groups) == 1:

        def record(states):
            name, inds = groups[0]
            return states[name][..., inds]

        return record

    inverse = as_index(np.argsort(order), device)

    def record(states):
        recs = [states[name][..., inds] for name, inds in groups]
        batch = torch.broadcast_shapes(*[r.shape[:-1] for r in recs])
        recs = [torch.broadcast_to(r, (*batch, r.shape[-1])) for r in recs]
        return torch.cat(recs, dim=-1)[..., inverse]

    return record


def build_init_and_step_fn(
    module: Module,
    voltage_solver: str = "jaxley.dhs",
    solver: str = "bwd_euler",
) -> Tuple[Callable, Callable]:
    """Return ``init_fn`` and ``step_fn`` which initialize modules and run update steps.

    This method can be used to gain additional control over the simulation workflow.
    It exposes a step function, which can be used to perform step-by-step updates
    of the differential equations.

    The structure of the module (channels, synapses, morphology) is read when the
    first step is taken. Do not modify the module structure afterwards (this is
    equivalent to `jit`-compiling the step function in Jaxley).

    Args:
        module: A ``jx.Module`` object that, for example a ``jx.Cell``.
        voltage_solver: Voltage solver used in step. Defaults to "jaxley.dhs".
        solver: ODE solver. One of
            { "bwd_euler" | "crank_nicolson" "| "fwd_euler" | "exp_euler" }.

    Returns:

        * ``init_fn(params, all_states=None, param_state=None, delta_t=0.025)``

          Callable which initializes the states and parameters.

          * Args:

            * ``params`` (list[dict]): returned by `.get_parameters()`.
            * ``all_states`` (dict | None = None): typically `None`.
            * ``param_state`` (list[dict] | None = None): returned by `.data_set()`.
            * ``delta_t`` (float = 0.025): the time step.

          * Returns:

            * ``all_states`` (dict).
            * ``all_params`` (dict), which can be passed to the `step_fn`.

        * ``step_fn(all_states, all_params, externals, external_inds, delta_t=0.025)``

          Callable which performs a single integration step.

          * Args:

            * ``all_states`` (dict): returned by `init_fn()`.
            * ``all_params`` (dict): returned by `init_fn()`.
            * ``externals`` (dict): obtained with `module.externals.copy()` but using
              only the external input at the current time step (see examples below).
            * ``external_inds`` (dict): obtained with `module.external_inds.copy()`.
            * ``delta_t`` (float): the time step.

          * Returns:

            * Updated ``all_states``  (dict).

    Example usage
    ^^^^^^^^^^^^^

    The following allows you to perform a step-by-step update of the differential
    equations.

    ::

        import torch
        import jaxley_pytorch as jx
        from jaxley_pytorch.integrate import build_init_and_step_fn

        t_max = 3.0
        delta_t = 0.025

        cell = jx.Cell()
        cell.record()
        cell.stimulate(jx.step_current(0, 1, 2, delta_t, t_max))
        params = cell.get_parameters()

        cell.to_torch()
        rec_inds = cell.recordings.rec_index.to_numpy()
        rec_states = cell.recordings.state.to_numpy()
        externals = cell.externals.copy()
        external_inds = cell.external_inds.copy()

        # Initialize.
        init_fn, step_fn = build_init_and_step_fn(cell)
        states, params = init_fn(params)
        recordings = [torch.stack(
            [states[rec_state][rec_ind] for rec_state, rec_ind in zip(rec_states, rec_inds)]
        )]

        # Loop over the ODE.
        steps = int(t_max / delta_t)  # Steps to integrate
        for step in range(steps):
            # Get externals at current timestep.
            externals_now = {}
            for key in externals.keys():
                externals_now[key] = externals[key][:, step]

            states = step_fn(
                states, params, externals_now, external_inds, delta_t=delta_t
            )
            recs = torch.stack(
                [states[rec_state][rec_ind] for rec_state, rec_ind in zip(rec_states, rec_inds)]
            )
            recordings.append(recs)

        rec = torch.stack(recordings, dim=0).T
    """
    # Initialize the external inputs and their indices.
    external_inds = module.external_inds.copy()
    voltage_solver = _resolve_voltage_solver(module, voltage_solver)
    cache: Dict[str, Any] = {}

    def init_fn(
        params: list[dict[str, Tensor]],
        all_states: dict | None = None,
        param_state: list[dict] | None = None,
        delta_t: float = 0.025,
    ) -> Tuple[Dict, Dict]:
        """Initializes the parameters and states of the neuron model.

        Args:
            params: List of trainable parameters.
            all_states: State if already initialized. Defaults to None.
            param_state: Parameters returned by `data_set`.. Defaults to None.
            delta_t: Step size. Defaults to 0.025.

        Returns:
            All states and parameters.
        """
        ref = first_float_tensor(params, param_state, all_states, module.externals)
        if ref is not None:
            module.to_torch(device=ref.device, dtype=ref.dtype)
        elif getattr(module, "torchnodes", None) is None:
            module.to_torch()
        all_states, all_params = _init(module, params, all_states, param_state, delta_t)
        return all_states, all_params

    def step_fn(
        all_states: Dict,
        all_params: Dict,
        externals: Dict,
        external_inds: Dict = external_inds,
        delta_t: float = 0.025,
    ) -> Dict:
        """Performs a single integration step with step size delta_t.

        Args:
            all_states: Current state of the neuron model.
            all_params: Current parameters of the neuron model.
            externals: External inputs.
            external_inds: External indices. Defaults to `module.external_inds`.
            delta_t: Time step. Defaults to 0.025.

        Returns:
            Updated states.
        """
        device = all_states["v"].device
        if cache.get("device") != device:
            cache["device"] = device
            cache["plan"] = module._get_step_plan(device, voltage_solver, solver)
        plan = cache["plan"]
        gathered = module._gather_step_params(plan, all_params)
        ext_inds = {k: as_index(v, device) for k, v in external_inds.items()}
        if "i" in externals:
            gathered["area_stim"] = all_params["area"][..., ext_inds["i"]]
        dtype = all_states["v"].dtype
        externals = {
            k: torch.as_tensor(v, dtype=dtype, device=device)
            for k, v in externals.items()
        }
        if solver == "exp_euler" and "exp_euler_transition" not in all_params:
            all_params = dict(all_params)
            all_params["exp_euler_transition"] = _exp_euler_matrix(
                module, all_params, delta_t
            )
        return module._step_with_plan(
            all_states,
            delta_t,
            ext_inds,
            externals,
            all_params,
            plan,
            gathered,
            solver=solver,
            voltage_solver=voltage_solver,
        )

    return init_fn, step_fn


def _init(module, params, all_states, param_state, delta_t, batch=None):
    """Return all states (expanded to the batch shape) and all parameters."""
    # Make the `trainable_params` of the same shape as the `param_state`, such that
    # they can be processed together by `get_all_parameters`.
    pstate = params_to_pstate(params, module.indices_set_by_trainables)
    if param_state is not None:
        pstate += param_state

    all_params = module.get_all_parameters(pstate)
    if all_states is None:
        all_states = module.get_all_states(pstate)
        if batch is None:
            batch = _batch_shape(
                list(tree_tensors(all_params)) + list(tree_tensors(all_states))
            )
        all_states = _expand_states(all_states, batch)
        all_states = module.append_channel_currents_to_states(
            all_states, all_params, delta_t
        )
    elif batch is not None:
        all_states = _expand_states(all_states, batch)
    return all_states, all_params


def _exp_euler_matrix(module, all_params, delta_t):
    exp_matrix = module.solver_customizers["exp_euler"]["exp_euler_transition"]
    if exp_matrix is None:
        exp_matrix = module.build_exp_euler_transition_matrix(
            delta_t=delta_t,
            axial_conductances=all_params["axial_conductances"],
        )
    return exp_matrix


def add_stimuli(
    externals: dict,
    external_inds: dict,
    data_stimuli: tuple[List[str], List[Tensor], List[np.ndarray]] | None = None,
) -> tuple[dict, dict]:
    """Extends the external inputs with the stimuli.

    Args:
        externals: Current external inputs.
        external_inds: Current external indices.
        data_stimuli: Additional data stimuli. Defaults to None.

    Returns:
        Updated external inputs and indices.
    """
    # If stimulus is inserted, add it to the external inputs.
    if "i" in externals.keys() or data_stimuli is not None:
        if "i" in externals.keys():
            if data_stimuli is not None:
                # Current stimuli can all be stacked because it's the same state var
                currents = [externals["i"]] + list(data_stimuli[1])
                externals["i"] = _cat_inputs(currents)
                all_inds = np.concatenate(
                    [np.asarray(external_inds["i"])]
                    + [np.asarray(i) for i in data_stimuli[2]]
                )
                external_inds["i"] = all_inds
        else:
            externals["i"] = _cat_inputs(list(data_stimuli[1]))
            external_inds["i"] = np.concatenate([np.asarray(i) for i in data_stimuli[2]])

    return externals, external_inds


def add_clamps(
    externals: dict,
    external_inds: dict,
    data_clamps: tuple[List[str], List[Tensor], List[np.ndarray]] | None = None,
) -> tuple[dict, dict]:
    """Adds clamps to the external inputs.

    Args:
        externals: Current external inputs.
        external_inds: Current external indices.
        data_clamps: Additional data clamps. Defaults to None.

    Returns:
        Updated external inputs and indices.
    """
    # If a clamp is inserted, add it to the external inputs.
    if data_clamps is not None:
        state_names, clamps, inds = data_clamps
        for name, clamp, ind in zip(state_names, clamps, inds):
            if name in externals.keys():
                externals[name] = _cat_inputs([externals[name], clamp])
                external_inds[name] = np.concatenate(
                    [np.asarray(external_inds[name]), np.asarray(ind)]
                )
            else:
                externals[name] = clamp
                external_inds[name] = np.asarray(ind)

    return externals, external_inds


def _cat_inputs(inputs: List[Tensor]) -> Tensor:
    """Concatenate inputs of shape `(..., num_inputs, time)` along `num_inputs`.

    Leading (batch) dimensions are broadcast. All inputs must have the same number of
    time steps.
    """
    ref = first_float_tensor(inputs)
    inputs = [
        torch.as_tensor(
            x,
            dtype=ref.dtype if ref is not None else None,
            device=ref.device if ref is not None else None,
        )
        for x in inputs
    ]
    batch = torch.broadcast_shapes(*[x.shape[:-2] for x in inputs])
    inputs = [torch.broadcast_to(x, (*batch, *x.shape[-2:])) for x in inputs]
    return torch.cat(inputs, dim=-2)


def integrate(
    module: Module,
    params: list[dict[str, Tensor]] = [],
    *,
    param_state: list[dict] | None = None,
    data_stimuli: tuple[List[str], List[Tensor], List[np.ndarray]] | None = None,
    data_clamps: tuple[List[str], List[Tensor], List[np.ndarray]] | None = None,
    t_max: float | None = None,
    delta_t: float = 0.025,
    solver: str = "bwd_euler",
    voltage_solver: str = "jaxley.dhs",
    checkpoint_lengths: Optional[List[int]] = None,
    all_states: Optional[Dict] = None,
    return_states: bool = False,
    compile: bool = False,
) -> Tensor:
    """Solves ODE and simulates neuron model.

    The simulation runs on the device (and with the dtype) of the trainable `params`,
    `param_state`, stimuli or `all_states` (whichever is given first). If none of them
    is a tensor, the default torch device and dtype are used.

    Every input may carry leading batch dimensions (this replaces `jax.vmap`):
    trainable parameters `(..., num_params)`, `data_set` values `(..., num_comps)`,
    stimuli and clamps `(..., num_inputs, time)`, and states `(..., num_comps)`. All
    batch dimensions are broadcast against each other. The recordings then have shape
    `(..., num_recordings, time)`.

    Args:
        params: Trainable parameters returned by `get_parameters()`.
        param_state: Parameters returned by `data_set`.
        data_stimuli: Outputs of `.data_stimulate()`, only needed if stimuli change
            across function calls.
        data_clamps: Outputs of `.data_clamp()`, only needed if clamps change across
            function calls.
        t_max: Duration of the simulation in milliseconds. If `t_max` is greater than
            the length of the stimulus input, the stimulus will be padded at the end
            with zeros. If `t_max` is smaller, then the stimulus with be truncated.
        delta_t: Time step of the solver in milliseconds.
        solver: Which ODE solver to use. Either of ["fwd_euler", "bwd_euler",
            "crank_nicolson", "exp_euler"]. Note that `exp_euler` is still
            experimental and is not yet optimized for networks.
        voltage_solver: Algorithm to solve quasi-tridiagonal linear system describing
            the voltage equations. The different options only take effect when
            `solver` is either `bwd_euler` or `crank_nicolson`. The options for
            `voltage_solver` are `jaxley.dhs` (and its variants `jaxley.dhs.cpu`,
            `jaxley.dhs.gpu`) and `jax.sparse` (a dense direct solve for testing, also
            available as `torch.dense`). For unbranched cables, we also support
            `jaxley.stone` (which is solved with the parallel DHS solver).
        checkpoint_lengths: Number of timesteps at every level of checkpointing. The
            `prod(checkpoint_lengths)` must be larger or equal to the desired number of
            simulated timesteps. Warning: the simulation is run for
            `prod(checkpoint_lengths)` timesteps, and the result is posthoc truncated
            to the desired simulation length. Therefore, a poor choice of
            `checkpoint_lengths` can lead to longer simulation time. If `None`, no
            checkpointing is applied. Checkpointing only takes effect if gradients are
            being recorded.
        all_states: An optional initial state that was returned by a previous
            `jx.integrate(..., return_states=True)` run. Overrides potentially
            trainable initial states.
        return_states: If True, it returns all states such that the current state of
            the `Module` can be set with `set_states`.
        compile: If True, the integration step is compiled with `torch.compile`
            (not part of Jaxley). The first call compiles (this can take a while);
            later calls with the same module structure, shapes, dtype, `delta_t` and
            solver reuse the compiled step.

    Example usage
    ^^^^^^^^^^^^^

    The most simple usage is the following:

    ::

        cell = jx.Cell()
        v = jx.integrate(cell, t_max=10.0)

    If ``t_max`` is not passed, then you must have inserted a stimulus, and ``t_max``
    will match the stimulus length.
    """
    voltage_solver = _resolve_voltage_solver(module, voltage_solver)

    assert module.initialized, "Module is not initialized, run `._initialize()`."

    # Infer the device and dtype of the simulation.
    ref = first_float_tensor(
        params,
        param_state,
        data_stimuli[1] if data_stimuli is not None else None,
        data_clamps[1] if data_clamps is not None else None,
        all_states,
        module.externals,
    )
    device = ref.device if ref is not None else torch.get_default_device()
    dtype = ref.dtype if ref is not None else torch.get_default_dtype()
    # Creates `.torchnodes` from `.nodes` and `.torchedges` from `.edges`.
    module.to_torch(device=device, dtype=dtype)

    # Initialize the external inputs and their indices.
    externals = {
        k: torch.as_tensor(v, dtype=dtype, device=device)
        for k, v in module.externals.items()
    }
    external_inds = module.external_inds.copy()

    # If stimulus is inserted, add it to the external inputs.
    externals, external_inds = add_stimuli(externals, external_inds, data_stimuli)

    # If a clamp is inserted, add it to the external inputs.
    externals, external_inds = add_clamps(externals, external_inds, data_clamps)

    if not externals.keys():
        # No stimulus was inserted and no clamp was set.
        assert t_max is not None, (
            "If no stimulus or clamp are inserted you have to specify the simulation"
            "duration at `jx.integrate(..., t_max=)`."
        )

    for key in externals.keys():
        externals[key] = torch.as_tensor(externals[key], dtype=dtype, device=device)
        # Shape `(time, ..., num_stimuli)`.
        externals[key] = torch.movedim(externals[key], -1, 0)

    if module.recordings.empty:
        raise ValueError("No recordings are set. Please set them.")

    # Shorten or pad stimulus depending on `t_max`.
    if t_max is not None:
        t_max_steps = int(t_max // delta_t + 1)

        # Pad or truncate the stimulus.
        for key in externals.keys():
            if t_max_steps > externals[key].shape[0]:
                if key == "i":
                    pad = torch.zeros(
                        (t_max_steps - externals["i"].shape[0], *externals["i"].shape[1:]),
                        dtype=dtype,
                        device=device,
                    )
                    externals["i"] = torch.cat((externals["i"], pad))
                else:
                    raise NotImplementedError(
                        "clamp must be at least as long as simulation."
                    )
            else:
                externals[key] = externals[key][:t_max_steps]

    # If necessary, pad the stimulus with zeros in order to simulate sufficiently long.
    # The total simulation length will be `prod(checkpoint_lengths)`. At the end, we
    # return only the first `nsteps_to_return` elements (plus the initial state).
    if externals:
        example_key = list(externals.keys())[0]
        nsteps_to_return = externals[example_key].shape[0]
    else:
        nsteps_to_return = t_max_steps

    if checkpoint_lengths is None:
        checkpoint_lengths = [nsteps_to_return]
        length = nsteps_to_return
    else:
        length = prod(checkpoint_lengths)
        size_difference = length - nsteps_to_return
        assert (
            nsteps_to_return <= length
        ), "The desired simulation duration is longer than `prod(nested_length)`."
        if externals:
            for key in externals.keys():
                dummy_external = torch.zeros(
                    (size_difference, *externals[key].shape[1:]),
                    dtype=dtype,
                    device=device,
                )
                externals[key] = torch.cat([externals[key], dummy_external])

    # Initialize parameters and states. All states get the full batch shape.
    pstate = params_to_pstate(params, module.indices_set_by_trainables)
    if param_state is not None:
        pstate = pstate + list(param_state)
    all_params = module.get_all_parameters(pstate)
    batch = _batch_shape(
        list(tree_tensors(all_params))
        + list(tree_tensors(all_states))
        + [torch.movedim(e, 0, -1)[..., 0] for e in externals.values()]
    )
    if all_states is None:
        init_states = module.get_all_states(pstate)
        batch = torch.broadcast_shapes(
            batch, _batch_shape(list(tree_tensors(init_states)))
        )
        init_states = _expand_states(init_states, batch)
    else:
        init_states = _expand_states(all_states, batch)

    plan = module._get_step_plan(device, voltage_solver, solver)
    step = module._get_compiled_step() if compile else module._step_with_plan
    gathered = module._gather_step_params(plan, all_params)
    if all_states is None:
        init_states = module._append_currents_with_plan(
            init_states, delta_t, plan, gathered
        )
    external_inds = {k: as_index(v, device) for k, v in external_inds.items()}
    if "i" in externals:
        gathered["area_stim"] = all_params["area"][..., external_inds["i"]]

    if solver == "exp_euler":
        all_params = dict(all_params)
        all_params["exp_euler_transition"] = _exp_euler_matrix(
            module, all_params, delta_t
        )

    record = _build_recorder(module, device)

    def _body_fun(state, externals_t):
        state = step(
            state,
            delta_t,
            external_inds,
            externals_t,
            all_params,
            plan,
            gathered,
            solver=solver,
            voltage_solver=voltage_solver,
        )
        return state, record(state)

    def _externals_at(t):
        return {key: val[t] for key, val in externals.items()}

    # Record the initial state.
    init_recording = record(init_states)

    # Run simulation. Checkpointing only matters if gradients are recorded.
    use_checkpointing = torch.is_grad_enabled() and len(checkpoint_lengths) > 1
    final_states, recordings = nested_checkpoint_scan(
        _body_fun,
        init_states,
        _externals_at,
        length=length,
        nested_lengths=checkpoint_lengths,
        use_checkpointing=use_checkpointing,
    )
    recs = torch.cat([init_recording[None], recordings[:nsteps_to_return]], dim=0)
    # Shape `(..., num_recordings, time)`.
    recs = torch.movedim(recs, 0, -1)
    return (recs, final_states) if return_states else recs
