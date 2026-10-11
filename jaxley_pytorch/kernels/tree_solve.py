# This file is part of jaxley-pytorch, a PyTorch port of Jaxley, a differentiable
# neuroscience simulator. Jaxley is licensed under the Apache License Version 2.0, see
# <https://www.apache.org/licenses/>
"""Fused GPU kernel (Triton) for tree-structured (Hines) linear systems.

Solves `A x = b` for a forest of trees with

    A[i, i] = diags[i],   A[p, c] = uppers[c],   A[c, p] = lowers[c]

for every edge between a child `c` and its parent `p`. The nodes must be ordered such
that every tree occupies a contiguous range of indices, its root comes first, and
every parent precedes its children (this holds for Jaxley's DHS solve order).

One Triton program solves one block of whole trees for one batch element, with the
block held in registers:

- Triangularization by height (leaves have height 0): at level `h`, every node of
  height `h` subtracts the contributions of its (already eliminated) children, which
  it gathers from padded child slots. No atomics are needed, so results are
  deterministic.
- Back-substitution by recursive doubling (as in Jaxley's `jaxley.dhs.gpu`):
  `x[c] = s[c] / d[c] - l[c] / d[c] * x[p]` is resolved in `log2(depth)` steps.

Compared to solving level by level with separate PyTorch operations, this replaces
O(levels) kernel launches per solve by a single launch.
"""
import math
from typing import Dict, Optional

import numpy as np
import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover
    HAS_TRITON = False

# Largest block (number of nodes of the trees solved by one program).
MAX_BLOCK = 8192


def build_forest_schedule(
    parent: np.ndarray, max_block: int = MAX_BLOCK
) -> Optional[Dict[str, np.ndarray]]:
    """Precompute the kernel schedule from the parent of every node (`-1` = root).

    Returns `None` if a single tree is larger than `max_block` or if the node order
    does not satisfy the requirements of the kernel.
    """
    parent = np.asarray(parent, dtype=np.int64)
    n = len(parent)
    roots = np.nonzero(parent < 0)[0]
    if n == 0 or len(roots) == 0 or roots[0] != 0:
        return None
    starts = roots
    ends = np.append(roots[1:], n)
    sizes = ends - starts
    if sizes.max() > max_block:
        return None

    # Every tree must be contiguous and parents must precede their children.
    tree_of = np.repeat(np.arange(len(starts)), sizes)
    idx = np.arange(n)
    nonroot = parent >= 0
    if not (
        np.all(parent[nonroot] >= starts[tree_of[nonroot]])
        and np.all(parent[nonroot] < idx[nonroot])
    ):
        return None

    # Depth (roots have depth 0); parents precede children.
    depth = np.zeros(n, dtype=np.int64)
    for i in np.nonzero(nonroot)[0]:
        depth[i] = depth[parent[i]] + 1

    # Height (leaves have height 0), deepest nodes first.
    height = np.zeros(n, dtype=np.int64)
    max_depth = int(depth.max())
    for dep in range(max_depth, 0, -1):
        nodes = np.nonzero(depth == dep)[0]
        np.maximum.at(height, parent[nodes], height[nodes] + 1)

    # Child slots: the k-th child of every node.
    children_nodes = np.nonzero(nonroot)[0]
    order = np.argsort(parent[children_nodes], kind="stable")
    children_nodes = children_nodes[order]
    child_parents = parent[children_nodes]
    first = np.searchsorted(child_parents, child_parents, side="left")
    slot = np.arange(len(children_nodes)) - first
    num_slots = int(slot.max()) + 1 if len(slot) > 0 else 1

    # Pack consecutive trees into blocks.
    blocks = []
    block_start, block_end = 0, 0
    for s, e in zip(starts, ends):
        if e - block_start > max_block:
            blocks.append((block_start, block_end))
            block_start = s
        block_end = e
    blocks.append((block_start, block_end))
    block_size = max(16, int(2 ** math.ceil(math.log2(max(e - s for s, e in blocks)))))

    nb = len(blocks)
    rel_parent = np.zeros((nb, block_size), dtype=np.int32)
    rel_height = -np.ones((nb, block_size), dtype=np.int32)
    rel_children = -np.ones((nb, num_slots, block_size), dtype=np.int32)
    block_info = np.zeros((nb, 4), dtype=np.int32)  # start, length, levels, doublings
    for b, (s, e) in enumerate(blocks):
        length = e - s
        local = np.arange(length)
        p = parent[s:e]
        rel_parent[b, :length] = np.where(p >= 0, p - s, local)
        rel_parent[b, length:] = np.arange(length, block_size)
        rel_height[b, :length] = height[s:e]
        in_block = (children_nodes >= s) & (children_nodes < e)
        rel_children[b, slot[in_block], child_parents[in_block] - s] = (
            children_nodes[in_block] - s
        )
        block_depth = int(depth[s:e].max())
        block_info[b] = [
            s,
            length,
            int(height[s:e].max()),
            int(math.ceil(math.log2(block_depth + 1))) if block_depth > 0 else 0,
        ]
    return {
        "parent": parent,
        "block_info": block_info,
        "rel_parent": rel_parent,
        "rel_height": rel_height,
        "rel_children": rel_children,
        "block_size": block_size,
        "num_slots": num_slots,
        "num_nodes": n,
    }



def build_level_schedule(
    parent: np.ndarray, block_nodes: int = 1024
) -> Optional[Dict[str, np.ndarray]]:
    """Schedule of the level kernel (`_tree_solve_levels_kernel`).

    Consecutive trees are packed into blocks of about `block_nodes` nodes (a single
    tree may be larger). Within a block, nodes are ordered by height, and for every
    height the range of positions is stored, so that every level touches only its own
    nodes. Returns `None` if the node order does not satisfy the kernel's requirements
    (as `build_forest_schedule`).
    """
    parent = np.asarray(parent, dtype=np.int64)
    n = len(parent)
    roots = np.nonzero(parent < 0)[0]
    if n == 0 or len(roots) == 0 or roots[0] != 0:
        return None
    starts = roots
    ends = np.append(roots[1:], n)
    sizes = ends - starts
    tree_of = np.repeat(np.arange(len(starts)), sizes)
    idx = np.arange(n)
    nonroot = parent >= 0
    if not (
        np.all(parent[nonroot] >= starts[tree_of[nonroot]])
        and np.all(parent[nonroot] < idx[nonroot])
    ):
        return None

    depth = np.zeros(n, dtype=np.int64)
    for i in np.nonzero(nonroot)[0]:
        depth[i] = depth[parent[i]] + 1
    height = np.zeros(n, dtype=np.int64)
    for dep in range(int(depth.max()), 0, -1):
        nodes = np.nonzero(depth == dep)[0]
        np.maximum.at(height, parent[nodes], height[nodes] + 1)

    children_nodes = np.nonzero(nonroot)[0]
    order = np.argsort(parent[children_nodes], kind="stable")
    children_nodes = children_nodes[order]
    child_parents = parent[children_nodes]
    first = np.searchsorted(child_parents, child_parents, side="left")
    slot = np.arange(len(children_nodes)) - first
    num_slots = int(slot.max()) + 1 if len(slot) > 0 else 1
    children = -np.ones((n, num_slots), dtype=np.int64)
    children[child_parents, slot] = children_nodes

    blocks = []
    block_start, block_end = 0, 0
    for s, e in zip(starts, ends):
        if e - block_start > block_nodes and block_end > block_start:
            blocks.append((block_start, block_end))
            block_start = s
        block_end = e
    blocks.append((block_start, block_end))

    ordered_nodes = np.zeros(n, dtype=np.int32)
    ordered_parent = -np.ones(n, dtype=np.int32)
    ordered_children = -np.ones((n, num_slots), dtype=np.int32)
    level_ptr = []
    block_info = np.zeros((len(blocks), 4), dtype=np.int32)  # start, length, levels, level_ptr offset
    for b, (s, e) in enumerate(blocks):
        h = height[s:e]
        order = np.argsort(h, kind="stable")
        ordered_nodes[s:e] = order
        p = parent[s:e][order]
        ordered_parent[s:e] = np.where(p >= 0, p - s, -1)
        c = children[s:e][order]
        ordered_children[s:e] = np.where(c >= 0, c - s, -1)
        levels = int(h.max())
        counts = np.bincount(h, minlength=levels + 1)
        block_info[b] = [s, e - s, levels, sum(len(x) for x in level_ptr)]
        level_ptr.append(np.concatenate([[0], np.cumsum(counts)]).astype(np.int32))
    return {
        "parent": parent,
        "block_info": block_info,
        "level_ptr": np.concatenate(level_ptr),
        "ordered_nodes": ordered_nodes,
        "ordered_parent": ordered_parent,
        "ordered_children": ordered_children.reshape(-1),
        "num_slots": num_slots,
        "num_nodes": n,
        "max_levels": int(block_info[:, 2].max()),
    }

if HAS_TRITON:

    @triton.jit
    def _tree_solve_kernel(
        D,
        L,
        U,
        S,
        X,
        row_stride,
        block_info,
        rel_parent,
        rel_height,
        rel_children,
        BLOCK: tl.constexpr,
        NUM_SLOTS: tl.constexpr,
    ):
        blk = tl.program_id(0)
        row = tl.program_id(1).to(tl.int64) * row_stride
        start = tl.load(block_info + blk * 4)
        length = tl.load(block_info + blk * 4 + 1)
        num_levels = tl.load(block_info + blk * 4 + 2)
        num_doublings = tl.load(block_info + blk * 4 + 3)

        offs = tl.arange(0, BLOCK)
        mask = offs < length
        ptr = row + start + offs
        d = tl.load(D + ptr, mask=mask, other=1.0)
        l = tl.load(L + ptr, mask=mask, other=0.0)
        u = tl.load(U + ptr, mask=mask, other=0.0)
        s = tl.load(S + ptr, mask=mask, other=0.0)
        meta = blk * BLOCK + offs
        height = tl.load(rel_height + meta)
        lu = l * u

        # Triangularization: eliminate children into their parents, by height.
        for level in range(1, num_levels + 1):
            factor = u / d
            alpha = lu / d
            beta = s * factor
            acc_d = tl.zeros([BLOCK], dtype=d.dtype)
            acc_s = tl.zeros([BLOCK], dtype=d.dtype)
            for k in tl.static_range(NUM_SLOTS):
                child = tl.load(rel_children + (blk * NUM_SLOTS + k) * BLOCK + offs)
                valid = child >= 0
                child = tl.where(valid, child, 0)
                acc_d += tl.where(valid, tl.gather(alpha, child, 0), 0.0)
                acc_s += tl.where(valid, tl.gather(beta, child, 0), 0.0)
            update = height == level
            d = tl.where(update, d - acc_d, d)
            s = tl.where(update, s - acc_s, s)

        # Back-substitution with recursive doubling.
        jump = tl.load(rel_parent + meta)
        is_root = jump == offs
        lower_effect = tl.where(is_root, 0.0, -l / d)
        solve_effect = s / d
        for _ in range(num_doublings):
            solve_effect = lower_effect * tl.gather(solve_effect, jump, 0) + solve_effect
            lower_effect = lower_effect * tl.gather(lower_effect, jump, 0)
            jump = tl.gather(jump, jump, 0)
        tl.store(X + ptr, solve_effect, mask=mask)


    @triton.jit
    def _tree_solve_levels_kernel(
        D,
        L,
        U,
        S,
        X,
        WD,
        row_stride,
        block_info,
        level_ptr,
        ordered_nodes,
        ordered_parent,
        ordered_children,
        W: tl.constexpr,
        NUM_SLOTS: tl.constexpr,
    ):
        """One block of trees for one batch element, level by level: every level
        processes only its own nodes (in chunks of `W`), with the partially eliminated
        diagonal (`WD`) and right-hand side (`X`) in global memory and a barrier
        between levels. Work is O(nodes) instead of O(levels * block), so this suits
        deep trees (reconstructed morphologies)."""
        blk = tl.program_id(0)
        row = tl.program_id(1).to(tl.int64) * row_stride
        start = tl.load(block_info + blk * 4)
        length = tl.load(block_info + blk * 4 + 1)
        num_levels = tl.load(block_info + blk * 4 + 2)
        lp = tl.load(block_info + blk * 4 + 3)
        base = row + start
        w = tl.arange(0, W)

        for c0 in range(0, length, W):
            i = c0 + w
            m = i < length
            tl.store(WD + base + i, tl.load(D + base + i, mask=m, other=1.0), mask=m)
            tl.store(X + base + i, tl.load(S + base + i, mask=m, other=0.0), mask=m)
        tl.debug_barrier()

        # Triangularization, leaves first: a node of height h eliminates its children
        # (which all have smaller heights).
        for h in range(1, num_levels + 1):
            lo = tl.load(level_ptr + lp + h)
            hi = tl.load(level_ptr + lp + h + 1)
            for c0 in range(lo, hi, W):
                pos = c0 + w
                m = pos < hi
                node = tl.load(ordered_nodes + start + pos, mask=m, other=0)
                acc_d = tl.zeros([W], dtype=WD.dtype.element_ty)
                acc_s = tl.zeros([W], dtype=WD.dtype.element_ty)
                for k in tl.static_range(NUM_SLOTS):
                    child = tl.load(ordered_children + (start + pos) * NUM_SLOTS + k, mask=m, other=-1)
                    valid = child >= 0
                    child = tl.where(valid, child, 0)
                    dc = tl.load(WD + base + child, mask=valid, other=1.0)
                    sc = tl.load(X + base + child, mask=valid, other=0.0)
                    lc = tl.load(L + base + child, mask=valid, other=0.0)
                    uc = tl.load(U + base + child, mask=valid, other=0.0)
                    factor = uc / dc
                    acc_d += tl.where(valid, lc * factor, 0.0)
                    acc_s += tl.where(valid, sc * factor, 0.0)
                dn = tl.load(WD + base + node, mask=m, other=1.0)
                sn = tl.load(X + base + node, mask=m, other=0.0)
                tl.store(WD + base + node, dn - acc_d, mask=m)
                tl.store(X + base + node, sn - acc_s, mask=m)
            tl.debug_barrier()

        # Back-substitution, roots first: parents have larger heights than their
        # children. `X` holds the eliminated right-hand side and is overwritten by the
        # solution.
        for hh in range(0, num_levels + 1):
            h = num_levels - hh
            lo = tl.load(level_ptr + lp + h)
            hi = tl.load(level_ptr + lp + h + 1)
            for c0 in range(lo, hi, W):
                pos = c0 + w
                m = pos < hi
                node = tl.load(ordered_nodes + start + pos, mask=m, other=0)
                par = tl.load(ordered_parent + start + pos, mask=m, other=-1)
                has_par = par >= 0
                xp = tl.load(X + base + tl.where(has_par, par, 0), mask=m & has_par, other=0.0)
                dn = tl.load(WD + base + node, mask=m, other=1.0)
                sn = tl.load(X + base + node, mask=m, other=0.0)
                ln = tl.load(L + base + node, mask=m, other=0.0)
                x = (sn - tl.where(has_par, ln * xp, 0.0)) / dn
                tl.store(X + base + node, x, mask=m)
            tl.debug_barrier()


class TreeSolveKernel:
    """Holds the schedule (on the GPU) and launches the fused tree solve."""

    def __init__(self, schedule: Dict[str, np.ndarray], device: torch.device):
        self.device = device
        self.block_size = schedule["block_size"]
        self.num_slots = schedule["num_slots"]
        self.num_nodes = schedule["num_nodes"]
        self.num_blocks = schedule["block_info"].shape[0]
        self.block_info = torch.as_tensor(schedule["block_info"], device=device)
        self.rel_parent = torch.as_tensor(schedule["rel_parent"], device=device)
        self.rel_height = torch.as_tensor(schedule["rel_height"], device=device)
        self.rel_children = torch.as_tensor(schedule["rel_children"], device=device)
        self.num_warps = int(min(16, max(4, self.block_size // 256)))
        parents = np.asarray(schedule["parent"], dtype=np.int64)
        self.has_parent = torch.as_tensor(parents >= 0, device=device)
        self.parent_safe = torch.as_tensor(np.where(parents >= 0, parents, 0), device=device)

    def solve(self, diags, lowers, uppers, solves):
        """Solve; all inputs have shape `(..., num_nodes)` (broadcast first)."""
        batch = diags.shape[:-1]
        n = diags.shape[-1]
        assert n == self.num_nodes
        d = diags.reshape(-1, n).contiguous()
        l = lowers.reshape(-1, n).contiguous()
        u = uppers.reshape(-1, n).contiguous()
        s = solves.reshape(-1, n).contiguous()
        x = torch.empty_like(s)
        grid = (self.num_blocks, d.shape[0])
        _tree_solve_kernel[grid](
            d,
            l,
            u,
            s,
            x,
            n,
            self.block_info,
            self.rel_parent,
            self.rel_height,
            self.rel_children,
            BLOCK=self.block_size,
            NUM_SLOTS=self.num_slots,
            num_warps=self.num_warps,
        )
        return x.reshape(*batch, n)

    def fused(self, diags, lowers, uppers, solves):
        """Differentiable solve (a custom op, opaque to `torch.compile`)."""
        return tree_solve_op(
            diags, lowers, uppers, solves, self.block_info, self.rel_parent,
            self.rel_height, self.rel_children, self.parent_safe, self.has_parent,
            self.block_size, self.num_slots, self.num_warps,
        )


class LevelTreeSolveKernel:
    """Holds the schedule of the level kernel (on the GPU) and launches it."""

    W = 128  # Nodes processed at once within a level.

    def __init__(self, schedule: Dict[str, np.ndarray], device: torch.device):
        self.device = device
        self.num_slots = schedule["num_slots"]
        self.num_nodes = schedule["num_nodes"]
        self.num_blocks = schedule["block_info"].shape[0]
        for key in ["block_info", "level_ptr", "ordered_nodes", "ordered_parent", "ordered_children"]:
            setattr(self, key, torch.as_tensor(schedule[key], device=device))
        parents = np.asarray(schedule["parent"], dtype=np.int64)
        self.has_parent = torch.as_tensor(parents >= 0, device=device)
        self.parent_safe = torch.as_tensor(np.where(parents >= 0, parents, 0), device=device)

    def solve(self, diags, lowers, uppers, solves):
        """Solve; all inputs have shape `(..., num_nodes)` (broadcast first)."""
        assert diags.shape[-1] == self.num_nodes
        return _launch_levels(
            diags, lowers, uppers, solves, self.block_info, self.level_ptr,
            self.ordered_nodes, self.ordered_parent, self.ordered_children,
            self.num_slots, self.W,
        )

    def fused(self, diags, lowers, uppers, solves):
        """Differentiable solve (a custom op, opaque to `torch.compile`)."""
        return tree_solve_levels_op(
            diags, lowers, uppers, solves, self.block_info, self.level_ptr,
            self.ordered_nodes, self.ordered_parent, self.ordered_children,
            self.parent_safe, self.has_parent, self.num_slots, self.W,
        )


def _launch_levels(diags, lowers, uppers, solves, block_info, level_ptr, ordered_nodes,
                   ordered_parent, ordered_children, num_slots, w):
    batch = diags.shape[:-1]
    n = diags.shape[-1]
    d = diags.reshape(-1, n).contiguous()
    l = lowers.reshape(-1, n).contiguous()
    u = uppers.reshape(-1, n).contiguous()
    s = solves.reshape(-1, n).contiguous()
    x = torch.empty_like(s)
    wd = torch.empty_like(d)
    grid = (block_info.shape[0], d.shape[0])
    _tree_solve_levels_kernel[grid](
        d, l, u, s, x, wd, n, block_info, level_ptr, ordered_nodes, ordered_parent,
        ordered_children, W=w, NUM_SLOTS=num_slots, num_warps=4,
    )
    return x.reshape(*batch, n)


def make_tree_solve_kernel(parent: np.ndarray, device: torch.device, mode: Optional[str] = None):
    """The fused solver for the forest given by `parent` (`None` if not supported).

    `mode` (default: environment variable `JAXLEY_PYTORCH_TREE_KERNEL`, else `auto`):
    `registers` holds a block of trees in registers and eliminates level by level over
    the whole block (fast for shallow trees); `levels` touches only the nodes of each
    level (fast for deep trees, e.g. reconstructed morphologies); `auto` estimates the
    work of both.
    """
    import os

    mode = mode or os.environ.get("JAXLEY_PYTORCH_TREE_KERNEL", "auto")
    forest = build_forest_schedule(parent) if mode in ["auto", "registers"] else None
    levels = build_level_schedule(parent) if mode in ["auto", "levels"] else None
    if mode == "auto" and forest is not None and levels is not None:
        # Work of the register kernel: every level visits the whole block; the level
        # kernel visits the nodes of a level (in chunks) and synchronizes once per level.
        register_work = int(np.sum(forest["block_info"][:, 2])) * forest["block_size"] * forest["num_slots"]
        chunks = 0
        for start, length, num_levels, lp in levels["block_info"]:
            widths = np.diff(levels["level_ptr"][lp : lp + num_levels + 2])
            chunks += int(np.sum(np.ceil(widths / LevelTreeSolveKernel.W)))
        level_work = chunks * LevelTreeSolveKernel.W * levels["num_slots"] * 4
        mode = "registers" if register_work <= level_work else "levels"
    if mode == "registers" and forest is not None:
        return TreeSolveKernel(forest, device)
    if mode in ["levels", "auto"] and levels is not None:
        return LevelTreeSolveKernel(levels, device)
    return None


if HAS_TRITON:

    @torch.library.custom_op("jaxley_pytorch::tree_solve", mutates_args=())
    def tree_solve_op(
        diags: torch.Tensor,
        lowers: torch.Tensor,
        uppers: torch.Tensor,
        solves: torch.Tensor,
        block_info: torch.Tensor,
        rel_parent: torch.Tensor,
        rel_height: torch.Tensor,
        rel_children: torch.Tensor,
        parent_safe: torch.Tensor,
        has_parent: torch.Tensor,
        block_size: int,
        num_slots: int,
        num_warps: int,
    ) -> torch.Tensor:
        """Fused tree solve as a custom op (opaque to `torch.compile`).

        `parent_safe` (the parent of every node, roots map to 0) and `has_parent` are
        only used by the backward pass. All four system tensors must have the same
        shape `(..., num_nodes)`.
        """
        batch = diags.shape[:-1]
        n = diags.shape[-1]
        d = diags.reshape(-1, n).contiguous()
        l = lowers.reshape(-1, n).contiguous()
        u = uppers.reshape(-1, n).contiguous()
        s = solves.reshape(-1, n).contiguous()
        x = torch.empty_like(s)
        grid = (block_info.shape[0], d.shape[0])
        _tree_solve_kernel[grid](
            d,
            l,
            u,
            s,
            x,
            n,
            block_info,
            rel_parent,
            rel_height,
            rel_children,
            BLOCK=block_size,
            NUM_SLOTS=num_slots,
            num_warps=num_warps,
        )
        return x.reshape(*batch, n)

    @tree_solve_op.register_fake
    def _tree_solve_fake(diags, lowers, uppers, solves, *args):
        return torch.empty_like(solves)

    def _tree_solve_setup_context(ctx, inputs, output):
        diags, lowers, uppers, solves = inputs[:4]
        schedule = inputs[4:10]
        ctx.save_for_backward(diags, lowers, uppers, output, *schedule)
        ctx.ints = inputs[10:]

    def _tree_solve_backward(ctx, grad_x):
        diags, lowers, uppers, x, *schedule = ctx.saved_tensors
        parent_safe, has_parent = schedule[4], schedule[5]
        # The transposed system has the same tree with `lowers` and `uppers` swapped.
        lam = tree_solve_op(diags, uppers, lowers, grad_x, *schedule, *ctx.ints)
        g_diags = -lam * x
        g_uppers = torch.where(has_parent, -lam[..., parent_safe] * x, 0.0)
        g_lowers = torch.where(has_parent, -lam * x[..., parent_safe], 0.0)
        return (g_diags, g_lowers, g_uppers, lam) + (None,) * 9

    tree_solve_op.register_autograd(
        _tree_solve_backward, setup_context=_tree_solve_setup_context
    )

    @torch.library.custom_op("jaxley_pytorch::tree_solve_levels", mutates_args=())
    def tree_solve_levels_op(
        diags: torch.Tensor,
        lowers: torch.Tensor,
        uppers: torch.Tensor,
        solves: torch.Tensor,
        block_info: torch.Tensor,
        level_ptr: torch.Tensor,
        ordered_nodes: torch.Tensor,
        ordered_parent: torch.Tensor,
        ordered_children: torch.Tensor,
        parent_safe: torch.Tensor,
        has_parent: torch.Tensor,
        num_slots: int,
        w: int,
    ) -> torch.Tensor:
        """Level-by-level tree solve as a custom op (see `tree_solve_op`)."""
        return _launch_levels(
            diags, lowers, uppers, solves, block_info, level_ptr, ordered_nodes,
            ordered_parent, ordered_children, num_slots, w,
        )

    @tree_solve_levels_op.register_fake
    def _tree_solve_levels_fake(diags, lowers, uppers, solves, *args):
        return torch.empty_like(solves)

    def _tree_solve_levels_setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs[:3], output, *inputs[4:11])
        ctx.ints = inputs[11:]

    def _tree_solve_levels_backward(ctx, grad_x):
        diags, lowers, uppers, x, *schedule = ctx.saved_tensors
        parent_safe, has_parent = schedule[5], schedule[6]
        lam = tree_solve_levels_op(diags, uppers, lowers, grad_x, *schedule, *ctx.ints)
        g_diags = -lam * x
        g_uppers = torch.where(has_parent, -lam[..., parent_safe] * x, 0.0)
        g_lowers = torch.where(has_parent, -lam * x[..., parent_safe], 0.0)
        return (g_diags, g_lowers, g_uppers, lam) + (None,) * 9

    tree_solve_levels_op.register_autograd(
        _tree_solve_levels_backward, setup_context=_tree_solve_levels_setup_context
    )


def fused_tree_solve(kernel, diags, lowers, uppers, solves):
    """Differentiable fused tree solve (inputs must have the same shape)."""
    return kernel.fused(diags, lowers, uppers, solves)
