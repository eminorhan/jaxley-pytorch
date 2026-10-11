# This file is part of jaxley-pytorch, a PyTorch port of Jaxley, a differentiable
# neuroscience simulator. Jaxley is licensed under the Apache License Version 2.0, see
# <https://www.apache.org/licenses/>
"""Voltage solvers.

All functions operate on tensors whose last dimension indexes nodes (compartments and
branchpoints) or compartment edges, with optional leading batch dimensions.
"""
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import Tensor

from jaxley_pytorch.solver_gate import exponential_euler


class DHSSolver:
    """Index tensors and the raw (non-differentiated) dendritic hierarchical solve.

    The tree-structured linear system `A x = b` has, for every edge between a child
    compartment `c` and its parent `p` (in solve order):

        A[p, c] = uppers[c],    A[c, p] = lowers[c],    A[i, i] = diags[i].

    The system is solved by triangularization (leaves first, one level of the tree at
    a time) followed by back-substitution (root first). The back-substitution is either
    sequential (`optimize_for_gpu=False`, Jaxley's `jaxley.dhs.cpu`) or uses recursive
    doubling (`optimize_for_gpu=True`, Jaxley's `jaxley.dhs.gpu`).
    """

    def __init__(
        self,
        solve_indexer: Dict[str, Any],
        optimize_for_gpu: bool,
        device,
        use_fused_kernel: bool = True,
    ):
        levels = [np.asarray(level, dtype=np.int64) for level in solve_indexer["node_order_grouped"]]
        levels = [level.reshape(-1, 2) for level in levels]
        self.device = device
        self.optimize_for_gpu = optimize_for_gpu
        # Root-first levels (for back-substitution) and leaves-first levels (for
        # triangularization). Each level is a tuple (children, parents).
        self.levels = [
            (
                torch.as_tensor(level[:, 0], device=device),
                torch.as_tensor(level[:, 1], device=device),
            )
            for level in levels
        ]
        self.flipped_levels = list(reversed(self.levels))
        self.steps = len(self.levels)
        self.all_children = torch.as_tensor(
            np.asarray(solve_indexer["all_children"], dtype=np.int64), device=device
        )
        self.all_parents = torch.as_tensor(
            np.asarray(solve_indexer["all_parents"], dtype=np.int64), device=device
        )
        self.num_recursive_steps = (
            int(np.ceil(np.log2(self.steps + 1))) if self.steps > 0 else 0
        )
        self.parent_lookup = torch.as_tensor(
            np.asarray(solve_indexer["parent_lookup"], dtype=np.int64), device=device
        )
        self._has_no_parent = self.parent_lookup < 0
        self._has_no_parent[0] = True  # The root (solve index 0).

        # On GPU, use the fused Triton kernel if the node order allows it.
        self.kernel = None
        if torch.device(device).type == "cuda" and use_fused_kernel:
            from jaxley_pytorch.kernels.tree_solve import HAS_TRITON, make_tree_solve_kernel

            if HAS_TRITON:
                parents = np.asarray(solve_indexer["parent_lookup"], dtype=np.int64)
                self.kernel = make_tree_solve_kernel(parents, torch.device(device))

    def raw_solve(
        self,
        diags: Tensor,
        lowers: Tensor,
        uppers: Tensor,
        solves: Tensor,
        use_kernel: bool = True,
    ):
        """Solve the tree-structured linear system (no autograd). On CUDA, this uses
        the fused kernel unless `use_kernel=False` (then it solves level by level)."""
        if use_kernel and self.kernel is not None and diags.is_cuda:
            return self.kernel.solve(diags, lowers, uppers, solves)
        d = diags.clone()
        s = solves.clone()

        # Triangularization, leaves first.
        for child, parent in self.flipped_levels:
            # Factor that the child row has to be multiplied by.
            multiplier = uppers[..., child] / d[..., child]
            d.index_add_(-1, parent, -lowers[..., child] * multiplier)
            s.index_add_(-1, parent, -s[..., child] * multiplier)

        if not self.optimize_for_gpu:
            # Sequential back-substitution, root first.
            lowers_norm = lowers / d
            s = s / d
            for child, parent in self.levels:
                s.index_add_(-1, child, -s[..., parent] * lowers_norm[..., child])
            return s
        else:
            return self._backsub_recursive_doubling(d, s, lowers)

    def _backsub_recursive_doubling(self, diags: Tensor, solves: Tensor, lowers: Tensor):
        """Backsubstitute with recursive doubling.

        The matrix describes a system like:
        diag[n] * x[n] + lower[n] * x[parent] = solve[n]

        We rephrase this as:
        x[n] = solve[n]/diag[n] - lower[n]/diag[n] * x[parent].
        x[n] = solve_effect[n] + lower_effect[n] * x[parent].

        Every iteration substitutes `x[parent]` into every equation, such that every
        node depends on its two-step, four-step, ... parent. Since
        `lower_effect[root] = 0`, the recursion terminates after `log2(levels)` steps.
        See `jaxley.solver_voltage._comp_based_backsub_recursive_doubling`.
        """
        # The `lowers` of nodes without parent (the root, `lowers[0]`, and the spurious
        # node that absorbs the root's parent lookup) have no meaning; they must be zero
        # for this recursion. (Jaxley zeros only `lowers[0]`; the spurious node's lower
        # is zero anyways, so the results are identical. Zeroing it explicitly makes
        # the solve independent of it, consistent with the backward pass.)
        lowers = lowers.masked_fill(self._has_no_parent, 0.0)

        lower_effect = -lowers / diags
        solve_effect = solves / diags

        parent_jump = self.parent_lookup
        for _ in range(self.num_recursive_steps):
            solve_effect = lower_effect * solve_effect[..., parent_jump] + solve_effect
            lower_effect = lower_effect * lower_effect[..., parent_jump]
            parent_jump = parent_jump[parent_jump]
        return solve_effect


class _DHSSolve(torch.autograd.Function):
    """Tree solve `x = A^{-1} b` with an efficient, differentiable backward pass.

    The backward pass is itself a solve with the transposed matrix, which has the same
    tree structure with `lowers` and `uppers` swapped:

        lambda = A^{-T} g,   dL/db = lambda,   dL/ddiag = -lambda * x,
        dL/dupper[c] = -lambda[p] * x[c],   dL/dlower[c] = -lambda[c] * x[p].

    This avoids storing the intermediate results of the O(levels) loop (the analogue of
    the custom JVP in Jaxley). Because the backward pass uses `_DHSSolve.apply`, it is
    differentiable itself (e.g. for Hessian-vector products).
    """

    @staticmethod
    def forward(diags, lowers, uppers, solves, solver):
        return solver.raw_solve(diags, lowers, uppers, solves)

    @staticmethod
    def setup_context(ctx, inputs, output):
        diags, lowers, uppers, solves, solver = inputs
        ctx.save_for_backward(diags, lowers, uppers, output)
        ctx.solver = solver

    @staticmethod
    def backward(ctx, grad_x):
        diags, lowers, uppers, x = ctx.saved_tensors
        solver = ctx.solver
        lam = _DHSSolve.apply(diags, uppers, lowers, grad_x, solver)
        children, parents = solver.all_children, solver.all_parents

        g_diags = g_lowers = g_uppers = g_solves = None
        if ctx.needs_input_grad[0]:
            g_diags = -lam * x
        if ctx.needs_input_grad[3]:
            g_solves = lam
        if ctx.needs_input_grad[2]:
            g_uppers = torch.zeros_like(lam).index_copy(
                -1, children, -lam[..., parents] * x[..., children]
            )
        if ctx.needs_input_grad[1]:
            g_lowers = torch.zeros_like(lam).index_copy(
                -1, children, -lam[..., children] * x[..., parents]
            )
        return g_diags, g_lowers, g_uppers, g_solves, None


class _DHSSolveForwardAD(_DHSSolve):
    """`_DHSSolve` with a forward-mode rule (used inside `torch.func.jvp` or
    `torch.autograd.forward_ad.dual_level`). The tangent is a solve with the same
    matrix: `dx = A^{-1} (db - dA x)`, where row `c` of `A` holds `lowers[c]` in the
    column of its parent `p` and row `p` holds `uppers[c]` in column `c`.

    Kept separate from `_DHSSolve` because `torch.compile` does not trace
    autograd.Functions with a custom `jvp`. It solves level by level also on CUDA:
    under `torch.func.jvp`, the operands that reach `forward` cannot be read by the
    Triton kernel (they have no storage).
    """

    @staticmethod
    def forward(diags, lowers, uppers, solves, solver):
        return solver.raw_solve(diags, lowers, uppers, solves, use_kernel=False)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _DHSSolve.setup_context(ctx, inputs, output)
        diags, lowers, uppers = inputs[:3]
        ctx.save_for_forward(diags, lowers, uppers, output)

    @staticmethod
    def jvp(ctx, t_diags, t_lowers, t_uppers, t_solves, _):
        diags, lowers, uppers, x = ctx.saved_tensors
        solver = ctx.solver
        children, parents = solver.all_children, solver.all_parents
        rhs = t_solves if t_solves is not None else torch.zeros_like(x)
        if t_diags is not None:
            rhs = rhs - t_diags * x
        if t_lowers is not None:
            rhs = rhs.index_add(-1, children, -t_lowers[..., children] * x[..., parents])
        if t_uppers is not None:
            rhs = rhs.index_add(-1, parents, -t_uppers[..., children] * x[..., children])
        return solver.raw_solve(diags, lowers, uppers, rhs, use_kernel=False)


def dhs_solve(diags, lowers, uppers, solves, solver: DHSSolver) -> Tensor:
    """Differentiable tree solve. All inputs must have the same shape.

    On GPU, this uses the fused Triton kernel (a custom op, which `torch.compile`
    treats as a single opaque call). Otherwise, it solves level by level. Inside a
    forward-mode AD level (e.g. `torch.func.jvp`), it uses `_DHSSolveForwardAD`
    (custom ops have no forward-mode rule).
    """
    if torch.autograd.forward_ad._current_level >= 0:
        return _DHSSolveForwardAD.apply(diags, lowers, uppers, solves, solver)
    if solver.kernel is not None and diags.is_cuda:
        from jaxley_pytorch.kernels.tree_solve import fused_tree_solve

        return fused_tree_solve(solver.kernel, diags, lowers, uppers, solves)
    return _DHSSolve.apply(diags, lowers, uppers, solves, solver)


def _append_spurious(x: Tensor, value: float) -> Tensor:
    """Append a spurious node (absorbs the `-1` padding of Jaxley's levels)."""
    pad = torch.full((*x.shape[:-1], 1), value, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=-1)


def _build_diags_and_solves(
    voltages,
    voltage_terms,
    constant_terms,
    axial_conductances,
    internal_node_inds,
    sinks,
    n_nodes,
    delta_t,
):
    """Build the diagonal and right-hand side of the implicit Euler system."""
    axial_conductances = delta_t * axial_conductances
    batch = torch.broadcast_shapes(
        voltages.shape[:-1],
        voltage_terms.shape[:-1],
        constant_terms.shape[:-1],
        axial_conductances.shape[:-1],
    )

    # Build diagonals.
    diags = torch.broadcast_to(delta_t * voltage_terms, (*batch, n_nodes))
    ones = torch.ones((1,), dtype=diags.dtype, device=diags.device)
    diags = diags.index_add(
        -1, internal_node_inds, ones.expand(*batch, len(internal_node_inds))
    )
    if len(sinks) > 0:
        diags = diags.index_add(
            -1,
            sinks,
            torch.broadcast_to(axial_conductances, (*batch, axial_conductances.shape[-1])),
        )

    # Build solve.
    vals = (
        voltages[..., internal_node_inds]
        + delta_t * constant_terms[..., internal_node_inds]
    )
    solves = torch.zeros((*batch, n_nodes), dtype=diags.dtype, device=diags.device)
    solves = solves.index_copy(
        -1, internal_node_inds, torch.broadcast_to(vals, (*batch, vals.shape[-1]))
    )
    return diags, solves, axial_conductances, batch


def step_voltage_implicit_with_dhs_solve(
    voltages,
    voltage_terms,
    constant_terms,
    axial_conductances,
    internal_node_inds,
    sinks,
    n_nodes,
    solve_indexer: Dict[str, Any],
    solver: DHSSolver,
    delta_t,
):
    """Return voltage update via compartment-based matrix inverse.

    Combined with an appropriate solve order, this results in `dendritic hierarchical
    scheduling` (DHS, Zhang et al., 2023). The solve order is defined in the graph
    backend of `Jaxley`.

    Args:
        solve_indexer: The `_dhs_solve_indexer` of the module, with the permutations
            to and from the solve order as `torch.long` tensors.
        solver: The `DHSSolver` built from the same indexer.
    """
    diags, solves, axial_conductances, batch = _build_diags_and_solves(
        voltages,
        voltage_terms,
        constant_terms,
        axial_conductances,
        internal_node_inds,
        sinks,
        n_nodes,
        delta_t,
    )

    # Why `n_nodes > 1`? For compartments (or point neurons), we save computation and
    # compile time by skipping the entire solve procedure.
    if axial_conductances.shape[-1] > 0:
        # Build lower and upper matrix.
        lowers_and_uppers = torch.broadcast_to(
            -axial_conductances, (*batch, axial_conductances.shape[-1])
        )

        # Reorder diagonals and solves.
        diags = diags[..., solve_indexer["map_to_solve_order"]]
        solves = solves[..., solve_indexer["map_to_solve_order"]]

        # Reorder the lower and upper values.
        lowers = lowers_and_uppers[..., solve_indexer["map_to_solve_order_lower"]]
        uppers = lowers_and_uppers[..., solve_indexer["map_to_solve_order_upper"]]

        # Add a spurious compartment that is modified by the masking.
        diags = _append_spurious(diags, 1.0)
        solves = _append_spurious(solves, 0.0)
        uppers = _append_spurious(uppers, 0.0)
        lowers = _append_spurious(lowers, 0.0)

        # Solve the voltage equations with an efficient custom backward pass.
        solution = dhs_solve(diags, lowers, uppers, solves, solver)

        # Remove the spurious compartment.
        solution = solution[..., :-1]
    else:
        solution = solves / diags

    solution = solution[..., solve_indexer["inv_map_to_solve_order"]]
    return solution


def step_voltage_implicit_with_dense_solve(
    voltages,
    voltage_terms,
    constant_terms,
    axial_conductances,
    internal_node_inds,
    sinks,
    sources,
    n_nodes,
    delta_t,
):
    """Implicit Euler step with a dense direct solve (`torch.linalg.solve`).

    This is the counterpart of Jaxley's `jax.sparse` voltage solver. It builds the full
    `(n_nodes, n_nodes)` matrix, so it is only meant for testing on small morphologies.
    """
    diags, solves, axial_conductances, batch = _build_diags_and_solves(
        voltages,
        voltage_terms,
        constant_terms,
        axial_conductances,
        internal_node_inds,
        sinks,
        n_nodes,
        delta_t,
    )
    # Row `sink`, column `source`: the axial conductance of a compartment edge acts on
    # the equation of its sink. (Jaxley passes CSC arrays with `row=source` to a CSR
    # solver, which amounts to the same matrix.)
    matrix = torch.diag_embed(diags)
    if len(sinks) > 0:
        vals = torch.broadcast_to(-axial_conductances, (*batch, len(sinks)))
        matrix = _batched_index_put(matrix, sinks, sources, vals)
    return torch.linalg.solve(matrix, solves[..., None])[..., 0]


def _batched_index_put(matrix: Tensor, rows: Tensor, cols: Tensor, vals: Tensor):
    """`matrix[..., rows, cols] += vals` for matrices with leading batch dimensions."""
    flat = matrix.reshape(-1, *matrix.shape[-2:])
    num = flat.shape[0]
    flat_vals = vals.reshape(num, vals.shape[-1])
    batch_inds = torch.arange(num, device=matrix.device)[:, None].expand(-1, len(rows))
    flat = flat.index_put(
        (batch_inds, rows[None, :].expand(num, -1), cols[None, :].expand(num, -1)),
        flat_vals,
        accumulate=True,
    )
    return flat.reshape(matrix.shape)


def step_voltage_explicit(
    voltages: Tensor,
    voltage_terms: Tensor,
    constant_terms: Tensor,
    axial_conductances: Tensor,
    sinks,
    sources,
    types,
    n_nodes,
    delta_t: float,
) -> Tensor:
    """Solve one timestep of branched nerve equations with explicit (forward) Euler."""
    update = _voltage_vectorfield(
        voltages,
        voltage_terms,
        constant_terms,
        axial_conductances,
        sinks,
        sources,
        types,
        n_nodes,
    )
    new_voltates = voltages + delta_t * update
    return new_voltates


def _voltage_vectorfield(
    voltages: Tensor,
    voltage_terms: Tensor,
    constant_terms: Tensor,
    axial_conductances: Tensor,
    sinks,
    sources,
    types,
    n_nodes,
) -> Tensor:
    """Evaluate the vectorfield of the nerve equation.

    `sinks`, `sources` are `torch.long` tensors and `types` is a numpy array (with the
    compartment edge type) of the same length.
    """
    batch = torch.broadcast_shapes(
        voltages.shape[:-1],
        voltage_terms.shape[:-1],
        constant_terms.shape[:-1],
        axial_conductances.shape[:-1],
    )
    voltages = torch.broadcast_to(voltages, (*batch, n_nodes))
    axial_conductances = torch.broadcast_to(
        axial_conductances, (*batch, axial_conductances.shape[-1])
    )
    # Membrane current update.
    vecfield = torch.broadcast_to(-voltage_terms * voltages + constant_terms, (*batch, n_nodes))

    # Compute branchpoint voltages. The voltages at branchpoints are a weighted
    # average of all neighboring compartments: vBP = (g1 v1 + g2 v2) / (g1 + g2)
    condition = np.isin(types, [3, 4])
    if np.sum(condition) > 0:
        cond_t = torch.as_tensor(np.nonzero(condition)[0], device=voltages.device)
        sink_comp_inds = sinks[cond_t]
        source_comp_inds = sources[cond_t]
        # Set set the voltages at branchpoints to zero.
        voltages = voltages.index_fill(-1, sink_comp_inds, 0.0)
        # Compute all terms that influence the branchpoints: g1 v1 + g2 v2
        voltages = voltages.index_add(
            -1,
            sink_comp_inds,
            axial_conductances[..., cond_t] * voltages[..., source_comp_inds],
        )
        # Compute the normalizer for the branchpoint: (g1 + g2)
        # Initialize at one such that we can also just divide the compartment voltages
        # by it.
        summed_axials = torch.ones(
            (*batch, n_nodes), dtype=voltages.dtype, device=voltages.device
        )
        summed_axials = summed_axials.index_fill(-1, sink_comp_inds, 0.0)
        summed_axials = summed_axials.index_add(
            -1, sink_comp_inds, axial_conductances[..., cond_t]
        )
        voltages = voltages / summed_axials

    # Update compartment voltages with the impact from other compartments and from
    # branchpoints.
    condition = np.isin(types, [0, 1, 2])
    if np.sum(condition) > 0:
        vecfield = vecfield.index_add(
            -1,
            sinks,
            (voltages[..., sources] - voltages[..., sinks]) * axial_conductances,
        )
    return vecfield


def step_voltage_exponential(
    voltages: Tensor,
    voltage_terms: Tensor,
    constant_terms: Tensor,
    axial_conductances: Tensor,
    matrix_exponential: Tensor,
    internal_node_inds: Tensor,
    delta_t: float,
) -> Tensor:
    """Solve one timestep of branched nerve equations with exponential Euler.

    This performs Lie-Trotter splitting for the voltages:

    dv/dt = G * v + g_channel * (E - V)

    v_{t+1} = exp(dt * G) * exp_euler(channel_terms, v_{t})

    I.e., with Lie-Trotter splitting, we first perform an update of the term related to
    the channels, and then update the 'diffusive' term G * v.

    Args:
        voltages: The voltages (including branchpoint voltages).
        voltage_terms: The linear part of the channels terms, i.e., g_channel.
        constant_terms: The constant part of the channel terms, i.e., g_channel * E
        axial_conductances: Unused (because we use the `matrix_exponential`).
        matrix_exponential: The matrix exp(G * dt), shape `(..., ncomp, ncomp)`.
        internal_node_inds: Indexes into all compartments.
        delta_t: Time step.

    Returns:
        The updated voltages (including branchpoint voltages).
    """
    # Update of the channels. This requires a `where` because: some compartments
    # might not have _any_ channels. In that case, their `voltate_terms = 0`, which
    # causes tau = inf. This if-case solves this.
    tau_is_inf = voltage_terms == 0
    v_tau_not_inf = exponential_euler(
        voltages, delta_t, constant_terms / voltage_terms, 1.0 / voltage_terms
    )
    v_tau_is_inf = voltages + constant_terms * delta_t
    voltages = torch.where(tau_is_inf, v_tau_is_inf, v_tau_not_inf)

    # Update of the diffusion matrix.
    internal = voltages[..., internal_node_inds]
    updated = torch.einsum("...ij,...j->...i", matrix_exponential, internal)
    batch = torch.broadcast_shapes(voltages.shape[:-1], updated.shape[:-1])
    voltages = torch.broadcast_to(voltages, (*batch, voltages.shape[-1]))
    return voltages.index_copy(
        -1, internal_node_inds, torch.broadcast_to(updated, (*batch, updated.shape[-1]))
    )
