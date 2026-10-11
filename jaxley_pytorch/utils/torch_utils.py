# This file is part of jaxley-pytorch, a PyTorch port of Jaxley, a differentiable
# neuroscience simulator. Jaxley is licensed under the Apache License Version 2.0, see
# <https://www.apache.org/licenses/>
"""Tensor helpers that replace the JAX idioms used by Jaxley.

Jaxley relies on `x.at[idx].set(val)`, `x.at[idx].add(val)` and `jax.vmap`. In this
port, every array that lives on compartments (or edges) is a tensor whose LAST dimension
indexes compartments (or edges), and which may carry any number of leading batch
dimensions. Batch dimensions replace `vmap`: they are broadcast against each other, so
batched parameters, stimuli or initial states can be mixed freely with unbatched ones.
"""

import math
from typing import Any, Callable, Iterator, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint


def infer_device() -> str:
    """Return the default flavour of the DHS voltage solver (`module._solver_device`).

    Jaxley uses the sequential solver (`cpu`, one node per level) on CPU and the
    level-parallel solver (`gpu`) on GPU, because XLA compiles the sequential loop into
    efficient code. Eagerly executed PyTorch pays per operation instead, so the
    level-parallel flavour is faster on every device (6x for an L5PC model on CPU). On
    GPU, the fused Triton kernel is used for either flavour. The sequential solver can
    still be chosen with `module._init_solver_jaxley_dhs_solve(allowed_nodes_per_level=1)`
    and `jx.integrate(..., voltage_solver="jaxley.dhs.cpu")`.

    Returns:
        `gpu`."""
    return "gpu"


# Device copies of index arrays, keyed by their content. Reusing them avoids
# host-to-device copies when a simulation is repeated (required for capturing it in a
# CUDA graph, and cheaper in general).
_INDEX_CACHE: dict = {}
_INDEX_CACHE_MAX = 4096


def as_index(idx: Any, device: torch.device) -> Tensor:
    """Return `idx` as a 1D or ND `torch.long` tensor on `device`.

    Non-tensor inputs (numpy arrays, lists) are converted once per content and device;
    later calls with the same content return the cached tensor (do not modify it).
    """
    if isinstance(idx, Tensor):
        return idx.to(device=device, dtype=torch.long)
    arr = np.ascontiguousarray(np.asarray(idx, dtype=np.int64))
    device = torch.device(device)
    key = (str(device), arr.shape, arr.tobytes())
    cached = _INDEX_CACHE.get(key)
    if cached is None:
        if len(_INDEX_CACHE) >= _INDEX_CACHE_MAX:
            _INDEX_CACHE.clear()
        cached = torch.as_tensor(arr.copy(), device=device)
        _INDEX_CACHE[key] = cached
    return cached


def as_tensor_like(val: Any, ref: Tensor) -> Tensor:
    """Return `val` as a tensor with the dtype and device of `ref`.

    Python and numpy scalars become a fill on `ref.device` rather than a
    host-to-device copy, so code that returns constants (e.g. a channel whose
    `compute_current` returns `0`) can run inside CUDA graph capture.
    """
    if isinstance(val, Tensor):
        return val.to(dtype=ref.dtype, device=ref.device)
    if isinstance(val, (int, float, bool, np.number, np.bool_)):
        return torch.full((), float(val), dtype=ref.dtype, device=ref.device)
    return torch.as_tensor(val, dtype=ref.dtype, device=ref.device)


def _normalize(idx: Tensor, n: int) -> Tensor:
    """Wrap negative indices (numpy / JAX semantics)."""
    return torch.remainder(idx, n) if n > 0 else idx


def set_at(x: Tensor, idx: Any, val: Any) -> Tensor:
    """Functional `x.at[..., idx].set(val)` along the last dimension of `x`.

    `idx` can be multi-dimensional (e.g. `(num_params, num_comps_per_param)`). `val`
    is broadcast against `(*batch, *idx.shape)`, where `batch` are the broadcast
    leading dimensions of `x` and `val`. Negative indices wrap around. If `idx`
    contains duplicates, which value is written is unspecified (as in JAX).
    """
    idx = as_index(idx, x.device)
    if idx.numel() == 0:
        return x
    n = x.shape[-1]
    val = torch.as_tensor(val, dtype=x.dtype, device=x.device)
    num_val_batch = max(val.ndim - idx.ndim, 0)
    batch = torch.broadcast_shapes(x.shape[:-1], val.shape[:num_val_batch])
    val = torch.broadcast_to(val, (*batch, *idx.shape)).reshape(*batch, -1)
    x = torch.broadcast_to(x, (*batch, n))
    return x.index_copy(-1, _normalize(idx.reshape(-1), n), val)


def add_at(x: Tensor, idx: Any, val: Any) -> Tensor:
    """Functional `x.at[..., idx].add(val)` along the last dimension (1D `idx`).

    Duplicate indices accumulate. Leading dimensions of `x` and `val` broadcast.
    """
    idx = as_index(idx, x.device)
    if idx.numel() == 0:
        return x
    n = x.shape[-1]
    val = torch.as_tensor(val, dtype=x.dtype, device=x.device)
    batch = torch.broadcast_shapes(x.shape[:-1], val.shape[:-1])
    val = torch.broadcast_to(val, (*batch, idx.shape[0]))
    x = torch.broadcast_to(x, (*batch, n))
    return x.index_add(-1, _normalize(idx, n), val)


def tree_tensors(tree: Any) -> Iterator[Tensor]:
    """Yield all tensors in a nested structure of lists, tuples and dicts."""
    if isinstance(tree, Tensor):
        yield tree
    elif isinstance(tree, dict):
        for v in tree.values():
            yield from tree_tensors(v)
    elif isinstance(tree, (list, tuple)):
        for v in tree:
            yield from tree_tensors(v)


def first_float_tensor(*trees: Any) -> Optional[Tensor]:
    """Return the first floating point tensor found in `trees` (or `None`)."""
    for tree in trees:
        for t in tree_tensors(tree):
            if t.is_floating_point():
                return t
    return None


def nested_checkpoint_scan(
    f: Callable[[Any, Any], Tuple[Any, Tensor]],
    init: Any,
    xs: Any,
    length: int,
    *,
    nested_lengths: Sequence[int],
    use_checkpointing: bool = True,
):
    """A Python-loop `scan` with (recursive) gradient checkpointing.

    Port of `jaxley.utils.jax_utils.nested_checkpoint_scan` (which follows
    https://github.com/google/jax/issues/2139). `f(carry, x_t) -> (carry, y_t)` is
    applied for every time step `t`; `xs` is a callable `xs(t) -> x_t` (or `None`).
    The outputs `y_t` are stacked along a new leading dimension.

    The outermost level is a plain loop over `nested_lengths[0]` chunks; every chunk
    is wrapped in a (non-reentrant) `torch.utils.checkpoint`, recursively. Gradient
    calculations therefore need memory O(sum(nested_lengths)) instead of
    O(prod(nested_lengths)) at the cost of `len(nested_lengths) - 1` extra forward
    passes. With a single nested length (or `use_checkpointing=False`) this is a plain
    loop.
    """
    if length != math.prod(nested_lengths):
        raise ValueError(f"inconsistent {length=} and {nested_lengths=}")
    # As in Jaxley, the outermost level is a plain loop over checkpointed chunks.
    return _scan_chunk(f, xs, use_checkpointing, init, 0, list(nested_lengths))


def _scan_chunk(f, xs, use_checkpointing, carry, start, lengths):
    """One (recursively checkpointed) chunk of `nested_checkpoint_scan`.

    A module-level function on purpose: a recursive closure refers to itself, and the
    resulting reference cycle would keep `f` and `xs` (e.g., the stimuli of a whole
    simulation) alive until the next garbage collection.
    """
    if len(lengths) == 1 or not use_checkpointing:
        ys = []
        for t in range(start, start + math.prod(lengths)):
            carry, y = f(carry, xs(t) if xs is not None else None)
            ys.append(y)
        return carry, torch.stack(ys)

    sub_length = math.prod(lengths[1:])
    ys = []
    for i in range(lengths[0]):
        carry, y = checkpoint(
            _scan_chunk,
            f,
            xs,
            use_checkpointing,
            carry,
            start + i * sub_length,
            lengths[1:],
            use_reentrant=False,
        )
        ys.append(y)
    return carry, torch.cat(ys)


def tree_map(fn: Callable, tree: Any, *rests: Any) -> Any:
    """Map `fn` over the leaves of nested lists, tuples and dicts (`jax.tree_map`).

    All trees in `rests` must have the same structure as `tree`. Anything that is not
    a list, tuple or dict is a leaf.
    """
    if isinstance(tree, dict):
        return {k: tree_map(fn, tree[k], *[r[k] for r in rests]) for k in tree}
    if isinstance(tree, (list, tuple)):
        mapped = [
            tree_map(fn, x, *[r[i] for r in rests]) for i, x in enumerate(tree)
        ]
        return type(tree)(mapped) if isinstance(tree, tuple) else mapped
    return fn(tree, *rests)


def tree_leaves(tree: Any) -> list:
    """Return the leaves of nested lists, tuples and dicts."""
    leaves = []
    tree_map(lambda x: leaves.append(x), tree)
    return leaves
