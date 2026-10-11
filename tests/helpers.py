# This file is part of jaxley-pytorch, a PyTorch port of Jaxley, a differentiable
# neuroscience simulator. Jaxley is licensed under the Apache License Version 2.0, see
# <https://www.apache.org/licenses/>

import numpy as np
import pandas as pd


def get_segment_xyzrL(section, comp_idx=None, loc=None, ncomp=8):
    assert (
        comp_idx is not None or loc is not None
    ), "Either comp_idx or loc must be provided."
    assert not (
        comp_idx is not None and loc is not None
    ), "Only one of comp_idx or loc can be provided."

    comp_len = 1 / ncomp
    loc = comp_len / 2 + comp_idx * comp_len if loc is None else loc

    n3d = section.n3d()
    x3d = np.array([section.x3d(i) for i in range(n3d)])
    y3d = np.array([section.y3d(i) for i in range(n3d)])
    z3d = np.array([section.z3d(i) for i in range(n3d)])
    L = np.array([section.arc3d(i) for i in range(n3d)])  # Cumulative arc lengths
    r3d = np.array([section.diam3d(i) / 2 for i in range(n3d)])
    if loc is None:
        return x3d, y3d, z3d, r3d
    else:
        total_length = L[-1]
        target_length = loc * total_length

        # Find segment containing target_length
        for i in range(1, n3d):
            if L[i] >= target_length:
                break
        else:
            i = n3d - 1

        # Interpolate between points i-1 and i
        L0, L1 = L[i - 1], L[i]
        t = (target_length - L0) / (L1 - L0)
        x = x3d[i - 1] + t * (x3d[i] - x3d[i - 1])
        y = y3d[i - 1] + t * (y3d[i] - y3d[i - 1])
        z = z3d[i - 1] + t * (z3d[i] - z3d[i - 1])
        r = r3d[i - 1] + t * (r3d[i] - r3d[i - 1])
        return x, y, z, r, L[-1] / ncomp


def jaxley2neuron_by_coords(jx_cell, neuron_secs, comp_idx=None, loc=None, ncomp=8):
    neuron_coords = {
        i: np.vstack(get_segment_xyzrL(sec, comp_idx=comp_idx, loc=loc, ncomp=ncomp))[
            :3
        ].T
        for i, sec in enumerate(neuron_secs)
    }
    neuron_coords = np.vstack(
        [np.hstack([k * np.ones((v.shape[0], 1)), v]) for k, v in neuron_coords.items()]
    )
    neuron_coords = pd.DataFrame(
        neuron_coords, columns=["global_branch_index", "x", "y", "z"]
    )
    neuron_coords["global_branch_index"] = neuron_coords["global_branch_index"].astype(
        int
    )

    neuron_loc_xyz = neuron_coords.groupby("global_branch_index").mean()
    jaxley_loc_xyz = (
        jx_cell.branch("all")
        .loc(loc)
        .nodes.set_index("global_branch_index")[["x", "y", "z"]]
    )

    jaxley2neuron_inds = {}
    for i, xyz in enumerate(jaxley_loc_xyz.to_numpy()):
        d = np.sqrt(((neuron_loc_xyz - xyz) ** 2)).sum(axis=1)
        jaxley2neuron_inds[i] = d.argmin()
    return jaxley2neuron_inds


def jaxley2neuron_by_group(
    jx_cell,
    neuron_secs,
    comp_idx=None,
    loc=None,
    ncomp=8,
    num_apical=20,
    num_tuft=20,
    num_basal=10,
):
    y_apical = (
        jx_cell.apical.nodes.groupby("global_branch_index")
        .mean()["y"]
        .abs()
        .sort_values()
    )
    trunk_inds = y_apical.index[:num_apical].tolist()
    tuft_inds = y_apical.index[-num_tuft:].tolist()
    basal_inds = (
        jx_cell.basal.nodes["global_branch_index"].unique()[:num_basal].tolist()
    )

    jaxley2neuron = jaxley2neuron_by_coords(
        jx_cell, neuron_secs, comp_idx=comp_idx, loc=loc, ncomp=ncomp
    )

    neuron_trunk_inds = [jaxley2neuron[i] for i in trunk_inds]
    neuron_tuft_inds = [jaxley2neuron[i] for i in tuft_inds]
    neuron_basal_inds = [jaxley2neuron[i] for i in basal_inds]

    neuron_inds = {
        "trunk": neuron_trunk_inds,
        "tuft": neuron_tuft_inds,
        "basal": neuron_basal_inds,
    }
    jaxley_inds = {"trunk": trunk_inds, "tuft": tuft_inds, "basal": basal_inds}
    return neuron_inds, jaxley_inds


def match_stim_loc(jx_cell, neuron_sec, comp_idx=None, loc=None, ncomp=8):
    stim_coords = get_segment_xyzrL(
        neuron_sec, comp_idx=comp_idx, loc=loc, ncomp=ncomp
    )[:3]
    stim_idx = (
        ((jx_cell.nodes[["x", "y", "z"]] - stim_coords) ** 2).sum(axis=1).argmin()
    )
    return stim_idx


def import_neuron_morph(fname, ncomp=8):
    from neuron import h

    _ = h.load_file("stdlib.hoc")
    _ = h.load_file("import3d.hoc")
    ncomp = 8

    ##################### NEURON ##################
    for sec in h.allsec():
        h.delete_section(sec=sec)

    cell = h.Import3d_SWC_read()
    cell.input(fname)
    i3d = h.Import3d_GUI(cell, False)
    i3d.instantiate(None)

    for sec in h.allsec():
        sec.nseg = ncomp
    return h, cell


def equal_both_nan_or_empty_df(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """Return whether all elements of two dataframes are identical."""
    if "xyzr" in a.columns:
        a = a.drop(columns="xyzr")
    if "xyzr" in b.columns:
        b = b.drop(columns="xyzr")
    if a.empty and b.empty:
        return True
    if set(a.columns) != set(b.columns):
        return False
    else:
        a = a[b.columns]

    equal = a.eq(b) | (a.isna() & b.isna())
    return bool(equal.to_numpy().all())


def value_and_grad(fn, argnums=0):
    """`jax.value_and_grad(fn, argnums)` for functions of pytrees of tensors.

    Every floating point tensor (or numpy array / float) in the differentiated
    arguments is replaced by a leaf tensor that requires grad. Unused inputs get zero
    gradients (as in JAX).
    """
    import numpy
    import torch

    from jaxley_pytorch.utils.torch_utils import tree_leaves, tree_map

    def _to_leaf(x):
        if isinstance(x, torch.Tensor) and x.is_floating_point():
            return x.detach().clone().requires_grad_(True)
        if isinstance(x, (float, numpy.ndarray, numpy.floating)):
            return torch.as_tensor(x, dtype=torch.get_default_dtype()).requires_grad_(
                True
            )
        return x

    def _is_leaf(x):
        return isinstance(x, torch.Tensor) and x.requires_grad

    def wrapped(*args, **kwargs):
        nums = (argnums,) if isinstance(argnums, int) else tuple(argnums)
        args = list(args)
        for i in nums:
            args[i] = tree_map(_to_leaf, args[i])
        value = fn(*args, **kwargs)
        all_grads = []
        for i in nums:
            leaves = [x for x in tree_leaves(args[i]) if _is_leaf(x)]
            grads = torch.autograd.grad(
                value, leaves, allow_unused=True, retain_graph=True
            )
            grads = iter(
                [torch.zeros_like(x) if g is None else g for x, g in zip(leaves, grads)]
            )
            all_grads.append(
                tree_map(lambda x: next(grads) if _is_leaf(x) else None, args[i])
            )
        grads = all_grads[0] if isinstance(argnums, int) else tuple(all_grads)
        return value.detach(), grads

    return wrapped


def grad(fn):
    """`jax.grad(fn)` (see `value_and_grad`)."""

    def wrapped(*args, **kwargs):
        return value_and_grad(fn)(*args, **kwargs)[1]

    return wrapped


def jit(fn=None, **kwargs):
    """Stand-in for `jax.jit`: PyTorch runs eagerly, so this returns `fn`."""
    if fn is None:
        return lambda f: f
    return fn


class _TorchFriendlyNumpy:
    """`numpy` whose functions accept torch tensors (converted to arrays first).

    The ported Jaxley tests compare simulation outputs with numpy functions
    (`np.allclose`, `np.max(np.abs(...))`, ...). Torch tensors are converted to numpy
    arrays before such calls. Tensors that require grad are refused (they must be
    handled with torch functions to keep their gradients).
    """

    def __init__(self, module):
        self._module = module

    @staticmethod
    def _convert(x):
        import torch

        if isinstance(x, torch.Tensor):
            if x.requires_grad:
                raise TypeError(
                    "A tensor that requires grad was passed to a numpy function. Use "
                    "the torch equivalent to keep its gradient."
                )
            return x.detach().cpu().numpy()
        if isinstance(x, list):
            return [_TorchFriendlyNumpy._convert(v) for v in x]
        if isinstance(x, tuple):
            return tuple(_TorchFriendlyNumpy._convert(v) for v in x)
        return x

    def __getattr__(self, name):
        attr = getattr(self._module, name)
        if callable(attr) and not isinstance(attr, type):
            convert = self._convert

            def wrapped(*args, **kwargs):
                args = [convert(a) for a in args]
                kwargs = {k: convert(v) for k, v in kwargs.items()}
                return attr(*args, **kwargs)

            return wrapped
        return attr


np = _TorchFriendlyNumpy(_np_module := __import__("numpy"))


def at_set(x, index, val):
    """Functional `x.at[index].set(val)` for tensors (test-side JAX idiom)."""
    import torch

    y = x.detach().clone()
    y[index] = torch.as_tensor(val, dtype=y.dtype, device=y.device)
    return y
