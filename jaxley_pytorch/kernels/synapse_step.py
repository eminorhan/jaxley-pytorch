# This file is part of jaxley-pytorch, a PyTorch port of Jaxley, a differentiable
# neuroscience simulator. Jaxley is licensed under the Apache License Version 2.0, see
# <https://www.apache.org/licenses/>
"""Fused GPU kernels (Triton) for the synapse step of regular layouts (not part of
Jaxley).

`Network._step_regular_synapses` simulates a synapse type whose synapses are ordered by
postsynaptic compartment, `k` per compartment. For the synapse models in `_models()`
(`KineticSynapse`, `SigmoidRateSynapse`), on CUDA and inside `jx.integrate`, the step of
such a synapse type is one custom op:

* forward: one program per postsynaptic compartment and block of batch elements loads
  the states and parameters of the compartment's synapses, gathers the presynaptic
  voltages, updates the states and sums the slope and offset of the currents (the
  finite difference of `Network._synapse_currents`), without atomic adds;
* backward: the same layout recomputes the forward quantities from the saved states
  (only the synapse states are kept for the backward pass) and writes the gradients of
  the states, of the parameters (summed over the batch) and of the presynaptic
  voltages. The latter go into a (synapse, batch element) buffer that a second kernel
  sums per presynaptic compartment with the transposed index: deterministic, without
  atomic adds (unless the transposed index would be much larger than the number of
  synapses, e.g. for a compartment with very many outgoing synapses).

The results agree with `Network._step_regular_synapses` to round-off
(`tests/test_synapse_kernels.py`). Parameters with batch dimensions or a trainable
area of the postsynaptic compartments use the PyTorch code instead.
`JAXLEY_PYTORCH_SYNAPSE_KERNEL=off` disables the kernels.
"""
import os
from typing import Dict, Optional

import numpy as np
import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover
    HAS_TRITON = False

KINETIC, SIGMOID_RATE = 0, 1
# Tiles (batch elements per program, warps) of the forward and the backward kernel,
# tuned for 1,024 x 100 synapses and 32 batch elements on an H100 (forward 28 us,
# backward 44 us, plus 8 us for the sum of the presynaptic gradients).
_FORWARD_TILE = (16, 8)
_BACKWARD_TILE = (8, 4)
# Synapses per program and loop iteration (at most).
_MAX_BLOCK_K = 128
# The transposed index (nodes x most outgoing synapses of a node) is used if it is at
# most this many times larger than the number of synapses, else atomic adds.
_MAX_TRANSPOSED_FACTOR = 4


def _models() -> Dict[type, tuple]:
    """Synapse class: (model, parameter names without the prefix)."""
    from jaxley_pytorch.synapses import KineticSynapse, SigmoidRateSynapse

    return {
        KineticSynapse: (KINETIC, ("gS", "e_syn", "alpha", "beta", "v_th", "delta")),
        SigmoidRateSynapse: (SIGMOID_RATE, ("gS", "x_offset", "slope")),
    }


def _transposed_index(pre: np.ndarray) -> tuple:
    """`(nodes, inv)`: the presynaptic nodes (sorted) and, as `(len(nodes), width)`
    int32, the synapses of every presynaptic node (padded with -1)."""
    nodes, local = np.unique(pre, return_inverse=True)
    counts = np.bincount(local, minlength=len(nodes))
    order = np.argsort(local, kind="stable")
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    rows = local[order]
    inv = np.full((len(nodes), max(int(counts.max()), 1)), -1, dtype=np.int32)
    inv[rows, np.arange(len(pre)) - starts[rows]] = order
    return nodes.astype(np.int32), inv


def kernel_plan(synapse_type, pre: np.ndarray, targets: np.ndarray, n_nodes: int, device) -> Dict:
    """Entries of the synapse plan for the fused kernels (empty if they do not apply)."""
    if (
        not HAS_TRITON
        or torch.device(device).type != "cuda"
        or os.environ.get("JAXLEY_PYTORCH_SYNAPSE_KERNEL", "auto") == "off"
    ):
        return {}
    entry = _models().get(type(synapse_type))
    if entry is None:
        return {}
    model, names = entry
    plan = {
        "kernel_model": model,
        "kernel_params": [f"{synapse_type._name}_{name}" for name in names],
        "kernel_pre": torch.as_tensor(pre.astype(np.int32), device=device),
        "kernel_targets": torch.as_tensor(targets.astype(np.int32), device=device),
    }
    nodes, inv = _transposed_index(pre)
    if inv.size > _MAX_TRANSPOSED_FACTOR * len(pre) + len(nodes):
        # Atomic adds for the gradient of the presynaptic voltages.
        inv = np.zeros((0, 1), dtype=np.int32)
    plan["kernel_pre_nodes"] = torch.as_tensor(nodes, device=device)
    plan["kernel_inv"] = torch.as_tensor(inv, device=device)
    return plan


def kernel_scalars(delta_t: float, diff: float, dtype, device) -> torch.Tensor:
    """The time step and the voltage step of the slope, as a tensor (filled on the
    device, without a host-to-device copy, so that it can be captured in a CUDA graph;
    `scalars[0] = delta_t` would copy)."""
    scalars = torch.full((2,), diff, dtype=dtype, device=device)
    scalars[:1].fill_(delta_t)
    return scalars


if HAS_TRITON:

    @triton.jit
    def _sum_over_batch(ptr, offset, values, mask, kmask):
        tl.store(ptr + offset, tl.sum(tl.where(mask, values, 0.0), axis=0), mask=kmask)

    @triton.jit
    def _forward_kernel(
        v_ptr, s_ptr, pre_ptr, tgt_ptr,
        p0_ptr, p1_ptr, p2_ptr, p3_ptr, p4_ptr, p5_ptr, area_ptr, scal_ptr,
        s_new_ptr, i_ptr, lin_ptr, const_ptr,
        B, N_NODES, E, T, K,
        MODEL: tl.constexpr, TARGETS_ARE_NODES: tl.constexpr, STORE_I: tl.constexpr,
        BLOCK_B: tl.constexpr, BLOCK_K: tl.constexpr, CHUNKS: tl.constexpr,
    ):
        target = tl.program_id(0)
        b = tl.program_id(1) * BLOCK_B + tl.arange(0, BLOCK_B)
        bmask = b < B
        b64 = b.to(tl.int64)
        dt = tl.load(scal_ptr)
        diff = tl.load(scal_ptr + 1)
        if TARGETS_ARE_NODES:
            node = target
        else:
            node = tl.load(tgt_ptr + target)
        vp = tl.load(v_ptr + b64 * N_NODES + node, mask=bmask, other=0.0)[:, None]
        lin = tl.zeros([BLOCK_B], dtype=v_ptr.dtype.element_ty)
        const = tl.zeros([BLOCK_B], dtype=v_ptr.dtype.element_ty)
        for chunk in tl.static_range(CHUNKS):  # Unrolled: `CHUNKS * BLOCK_K >= K`.
            kk = chunk * BLOCK_K + tl.arange(0, BLOCK_K)
            kmask = kk < K
            m2 = bmask[:, None] & kmask[None, :]
            syn = target * K + kk  # int32: fewer than 2^31 synapses.
            pre = tl.load(pre_ptr + syn, mask=kmask, other=0)
            vpre = tl.load(v_ptr + b64[:, None] * N_NODES + pre[None, :], mask=m2, other=0.0)
            area = tl.load(area_ptr + syn, mask=kmask, other=1.0)[None, :]
            gs = tl.load(p0_ptr + syn, mask=kmask, other=0.0)[None, :]
            offs = b64[:, None] * E + syn[None, :]
            if MODEL == 0:
                e_syn = tl.load(p1_ptr + syn, mask=kmask, other=0.0)[None, :]
                alpha = tl.load(p2_ptr + syn, mask=kmask, other=0.0)[None, :]
                beta = tl.load(p3_ptr + syn, mask=kmask, other=1.0)[None, :]
                v_th = tl.load(p4_ptr + syn, mask=kmask, other=0.0)[None, :]
                delta = tl.load(p5_ptr + syn, mask=kmask, other=1.0)[None, :]
                s = tl.load(s_ptr + offs, mask=m2, other=0.0)
                transmitter = tl.sigmoid((vpre - v_th) / delta)
                a = alpha * transmitter
                rate = a + beta
                s_inf = a / rate
                s_new = s_inf + (s - s_inf) * tl.exp(-dt * rate)
                tl.store(s_new_ptr + offs, s_new, mask=m2)
                i0 = gs * s_new * (vp - e_syn)
                i1 = gs * s_new * ((vp + diff) - e_syn)
            else:
                x_offset = tl.load(p1_ptr + syn, mask=kmask, other=0.0)[None, :]
                slope = tl.load(p2_ptr + syn, mask=kmask, other=0.0)[None, :]
                i0 = -gs * tl.sigmoid((vpre - x_offset) * slope)
                i1 = -gs * tl.sigmoid(((vpre + diff) - x_offset) * slope)
            d0 = i0 / area * 100000.0
            d1 = i1 / area * 100000.0
            vt = (d1 - d0) / diff
            ct = d0 - vt * vp
            if STORE_I:
                tl.store(i_ptr + offs, i0, mask=m2)
            lin += tl.sum(tl.where(m2, vt, 0.0), axis=1)
            const += tl.sum(tl.where(m2, ct, 0.0), axis=1)
        tl.store(lin_ptr + b64 * T + target, lin, mask=bmask)
        tl.store(const_ptr + b64 * T + target, const, mask=bmask)

    @triton.jit
    def _backward_kernel(
        v_ptr, s_ptr, pre_ptr, tgt_ptr,
        p0_ptr, p1_ptr, p2_ptr, p3_ptr, p4_ptr, p5_ptr, area_ptr, scal_ptr,
        gsn_ptr, gi_ptr, glin_ptr, gconst_ptr,
        gs_ptr, gpre_ptr, gpost_ptr, gparam_ptr,
        B, N_NODES, E, T, K, NB,
        MODEL: tl.constexpr, TARGETS_ARE_NODES: tl.constexpr, HAS_GSN: tl.constexpr,
        HAS_GI: tl.constexpr, NEED: tl.constexpr, R1: tl.constexpr, R2: tl.constexpr,
        R3: tl.constexpr, R4: tl.constexpr, R5: tl.constexpr, ATOMIC: tl.constexpr,
        BLOCK_B: tl.constexpr, BLOCK_K: tl.constexpr, CHUNKS: tl.constexpr,
    ):
        target = tl.program_id(0)
        pid_b = tl.program_id(1)
        b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        bmask = b < B
        b64 = b.to(tl.int64)
        dt = tl.load(scal_ptr)
        diff = tl.load(scal_ptr + 1)
        if TARGETS_ARE_NODES:
            node = target
        else:
            node = tl.load(tgt_ptr + target)
        vp = tl.load(v_ptr + b64 * N_NODES + node, mask=bmask, other=0.0)[:, None]
        glin = tl.load(glin_ptr + b64 * T + target, mask=bmask, other=0.0)[:, None]
        gconst = tl.load(gconst_ptr + b64 * T + target, mask=bmask, other=0.0)[:, None]
        g_vp_sum = tl.zeros([BLOCK_B], dtype=v_ptr.dtype.element_ty)
        # The gradients of the needed parameters (`NEED`) are rows `R1`... of
        # `gparam_ptr` (`(rows, NB, E)`; the first needed parameter is row 0).
        param = pid_b.to(tl.int64) * E  # Offset of this block's parameter gradients.
        stride = (pid_b.to(tl.int64) * 0 + NB) * E  # From one row to the next (int64).
        for chunk in tl.static_range(CHUNKS):  # Unrolled: `CHUNKS * BLOCK_K >= K`.
            kk = chunk * BLOCK_K + tl.arange(0, BLOCK_K)
            kmask = kk < K
            m2 = bmask[:, None] & kmask[None, :]
            syn = target * K + kk  # int32: fewer than 2^31 synapses.
            pre = tl.load(pre_ptr + syn, mask=kmask, other=0)
            vpre = tl.load(v_ptr + b64[:, None] * N_NODES + pre[None, :], mask=m2, other=0.0)
            area = tl.load(area_ptr + syn, mask=kmask, other=1.0)[None, :]
            gs = tl.load(p0_ptr + syn, mask=kmask, other=0.0)[None, :]
            offs = b64[:, None] * E + syn[None, :]
            # The forward quantities.
            if MODEL == 0:
                e_syn = tl.load(p1_ptr + syn, mask=kmask, other=0.0)[None, :]
                alpha = tl.load(p2_ptr + syn, mask=kmask, other=0.0)[None, :]
                beta = tl.load(p3_ptr + syn, mask=kmask, other=1.0)[None, :]
                v_th = tl.load(p4_ptr + syn, mask=kmask, other=0.0)[None, :]
                delta = tl.load(p5_ptr + syn, mask=kmask, other=1.0)[None, :]
                s = tl.load(s_ptr + offs, mask=m2, other=0.0)
                x = (vpre - v_th) / delta
                transmitter = tl.sigmoid(x)
                a = alpha * transmitter
                rate = a + beta
                s_inf = a / rate
                decay = tl.exp(-dt * rate)
                s_new = s_inf + (s - s_inf) * decay
                w0 = vp - e_syn
                w1 = (vp + diff) - e_syn
                i0 = gs * s_new * w0
                i1 = gs * s_new * w1
            else:
                x_offset = tl.load(p1_ptr + syn, mask=kmask, other=0.0)[None, :]
                slope = tl.load(p2_ptr + syn, mask=kmask, other=0.0)[None, :]
                u0 = vpre - x_offset
                u1 = (vpre + diff) - x_offset
                sig0 = tl.sigmoid(u0 * slope)
                sig1 = tl.sigmoid(u1 * slope)
                i0 = -gs * sig0
                i1 = -gs * sig1
            d0 = i0 / area * 100000.0
            d1 = i1 / area * 100000.0
            vt = (d1 - d0) / diff
            # Backward of `lin = sum(vt)`, `const = sum(d0 - vt * vp)`, `d = i / area * 1e5`.
            g_vt = glin - gconst * vp
            g_vp = -gconst * vt
            g_d1 = g_vt / diff
            g_d0 = gconst - g_d1
            g_i0 = g_d0 * 100000.0 / area
            g_i1 = g_d1 * 100000.0 / area
            if HAS_GI:
                g_i0 += tl.load(gi_ptr + offs, mask=m2, other=0.0)
            if MODEL == 0:
                g_w = g_i0 * w0 + g_i1 * w1
                g_sn = gs * g_w
                if HAS_GSN:
                    g_sn += tl.load(gsn_ptr + offs, mask=m2, other=0.0)
                g_vp += gs * s_new * (g_i0 + g_i1)
                g_sinf = g_sn * (1.0 - decay)
                g_rate = -dt * decay * g_sn * (s - s_inf) - g_sinf * s_inf / rate
                g_a = g_sinf / rate + g_rate
                g_x = g_a * alpha * transmitter * (1.0 - transmitter)
                g_vpre = g_x / delta
                tl.store(gs_ptr + offs, g_sn * decay, mask=m2)
                if NEED & 1:  # gS
                    _sum_over_batch(gparam_ptr, param + syn, s_new * g_w, m2, kmask)
                if NEED & 2:  # e_syn
                    _sum_over_batch(gparam_ptr, R1 * stride + param + syn, -gs * s_new * (g_i0 + g_i1), m2, kmask)
                if NEED & 4:  # alpha
                    _sum_over_batch(gparam_ptr, R2 * stride + param + syn, g_a * transmitter, m2, kmask)
                if NEED & 8:  # beta
                    _sum_over_batch(gparam_ptr, R3 * stride + param + syn, g_rate, m2, kmask)
                if NEED & 16:  # v_th
                    _sum_over_batch(gparam_ptr, R4 * stride + param + syn, -g_vpre, m2, kmask)
                if NEED & 32:  # delta
                    _sum_over_batch(gparam_ptr, R5 * stride + param + syn, -g_vpre * x, m2, kmask)
            else:
                g_z0 = -gs * g_i0 * sig0 * (1.0 - sig0)
                g_z1 = -gs * g_i1 * sig1 * (1.0 - sig1)
                g_vpre = (g_z0 + g_z1) * slope
                if NEED & 1:  # gS
                    _sum_over_batch(gparam_ptr, param + syn, -(g_i0 * sig0 + g_i1 * sig1), m2, kmask)
                if NEED & 2:  # x_offset
                    _sum_over_batch(gparam_ptr, R1 * stride + param + syn, -g_vpre, m2, kmask)
                if NEED & 4:  # slope
                    _sum_over_batch(gparam_ptr, R2 * stride + param + syn, g_z0 * u0 + g_z1 * u1, m2, kmask)
            g_vp_sum += tl.sum(tl.where(m2, g_vp, 0.0), axis=1)
            g_vpre = tl.where(m2, g_vpre, 0.0)
            if ATOMIC:
                tl.atomic_add(gpre_ptr + b64[:, None] * N_NODES + pre[None, :], g_vpre, mask=m2)
            else:
                tl.store(gpre_ptr + syn[None, :].to(tl.int64) * B + b64[:, None], g_vpre, mask=m2)
        tl.store(gpost_ptr + b64 * T + target, g_vp_sum, mask=bmask)

    @triton.jit
    def _presynaptic_sum_kernel(
        gpre_ptr, inv_ptr, nodes_ptr, gv_ptr, B, N_NODES, D, BLOCK_B: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """`gv[b, nodes[j]] = sum_d gpre[inv[j, d], b]` over the valid (>= 0) `inv`, for
        every presynaptic node `nodes[j]`."""
        j = tl.program_id(0)
        node = tl.load(nodes_ptr + j)
        b = tl.program_id(1) * BLOCK_B + tl.arange(0, BLOCK_B)
        bmask = b < B
        acc = tl.zeros([BLOCK_B], dtype=gv_ptr.dtype.element_ty)
        for start in range(0, D, BLOCK_D):
            d = start + tl.arange(0, BLOCK_D)
            idx = tl.load(inv_ptr + j.to(tl.int64) * D + d, mask=d < D, other=-1)
            valid = idx >= 0
            rows = tl.where(valid, idx, 0).to(tl.int64)
            vals = tl.load(
                gpre_ptr + rows[:, None] * B + b[None, :],
                mask=valid[:, None] & bmask[None, :],
                other=0.0,
            )
            acc += tl.sum(vals, axis=0)
        tl.store(gv_ptr + b.to(tl.int64) * N_NODES + node, acc, mask=bmask)

    def _tiles(tile, B: int, k: int):
        block_b, warps = tile
        return min(triton.next_power_of_2(B), block_b), min(triton.next_power_of_2(k), _MAX_BLOCK_K), warps

    @torch.library.custom_op("jaxley_pytorch::synapse_step", mutates_args=())
    def synapse_step_op(
        v: torch.Tensor, s: torch.Tensor, pre: torch.Tensor, targets: torch.Tensor,
        p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor,
        p4: torch.Tensor, p5: torch.Tensor, area: torch.Tensor, scalars: torch.Tensor,
        inv: torch.Tensor, pre_nodes: torch.Tensor, k: int, model: int,
        targets_are_nodes: bool, store_current: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One step of the synapses of one type: `(states, current through every
        synapse (if `store_current`), slope, offset)`, shapes `(B, E)`, `(B, E)`,
        `(B, T)`, `(B, T)` for `v` of shape `(B, nodes)` and `E = T * k` synapses."""
        B, n = v.shape
        E = pre.shape[0]
        T = E // k
        block_b, block_k, warps = _tiles(_FORWARD_TILE, B, k)
        kinetic = model == KINETIC
        s_new = torch.empty_like(s) if kinetic else v.new_empty(0)
        current = v.new_empty(B, E) if store_current else v.new_empty(0)
        lin = v.new_empty(B, T)
        const = v.new_empty(B, T)
        _forward_kernel[(T, triton.cdiv(B, block_b))](
            v, s if kinetic else v, pre, targets, p0, p1, p2, p3, p4, p5, area, scalars,
            s_new if kinetic else v, current if store_current else v, lin, const,
            B, n, E, T, k, MODEL=model, TARGETS_ARE_NODES=targets_are_nodes,
            STORE_I=store_current, BLOCK_B=block_b, BLOCK_K=block_k,
            CHUNKS=triton.cdiv(k, block_k), num_warps=warps,
        )
        return s_new, current, lin, const

    @synapse_step_op.register_fake
    def _synapse_step_fake(v, s, pre, targets, p0, p1, p2, p3, p4, p5, area, scalars, inv, pre_nodes, k, model, targets_are_nodes, store_current):
        B, E = v.shape[0], pre.shape[0]
        s_new = torch.empty_like(s) if model == KINETIC else v.new_empty(0)
        current = v.new_empty(B, E) if store_current else v.new_empty(0)
        return s_new, current, v.new_empty(B, E // k), v.new_empty(B, E // k)

    @torch.library.custom_op("jaxley_pytorch::synapse_step_backward", mutates_args=())
    def synapse_step_backward_op(
        v: torch.Tensor, s: torch.Tensor, pre: torch.Tensor, targets: torch.Tensor,
        p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor,
        p4: torch.Tensor, p5: torch.Tensor, area: torch.Tensor, scalars: torch.Tensor,
        inv: torch.Tensor, pre_nodes: torch.Tensor, g_s_new: torch.Tensor,
        g_current: torch.Tensor, g_lin: torch.Tensor, g_const: torch.Tensor, k: int,
        model: int, targets_are_nodes: bool, has_g_s_new: bool, has_g_current: bool,
        need: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gradients of `v`, of the states and of the parameters in the bit mask `need`
        (one row each, in the order of the parameters)."""
        B, n = v.shape
        E = pre.shape[0]
        T = E // k
        block_b, block_k, warps = _tiles(_BACKWARD_TILE, B, k)
        nb = triton.cdiv(B, block_b)
        kinetic = model == KINETIC
        atomic = inv.shape[0] == 0
        g_s = torch.empty_like(s) if kinetic else v.new_empty(0)
        g_post = v.new_empty(B, T)
        rows = [bin(need & ((1 << i) - 1)).count("1") for i in range(6)]
        g_params = v.new_empty(bin(need).count("1"), nb, E)
        g_v = torch.zeros_like(v)
        g_pre = g_v if atomic else v.new_empty(E, B)
        _backward_kernel[(T, nb)](
            v, s if kinetic else v, pre, targets, p0, p1, p2, p3, p4, p5, area, scalars,
            g_s_new if has_g_s_new else v, g_current if has_g_current else v, g_lin, g_const,
            g_s if kinetic else v, g_pre, g_post, g_params if need else v, B, n, E, T, k, nb,
            MODEL=model, TARGETS_ARE_NODES=targets_are_nodes, HAS_GSN=has_g_s_new,
            HAS_GI=has_g_current, NEED=need, R1=rows[1], R2=rows[2], R3=rows[3], R4=rows[4],
            R5=rows[5], ATOMIC=atomic, BLOCK_B=block_b, BLOCK_K=block_k,
            CHUNKS=triton.cdiv(k, block_k), num_warps=warps,
        )
        if not atomic:
            block_b2 = min(triton.next_power_of_2(B), 64)
            _presynaptic_sum_kernel[(pre_nodes.shape[0], triton.cdiv(B, block_b2))](
                g_pre, inv, pre_nodes, g_v, B, n, inv.shape[1], BLOCK_B=block_b2, BLOCK_D=32,
                num_warps=4,
            )
        if targets_are_nodes:
            g_v = g_v + g_post
        else:
            g_v = g_v.index_add(-1, targets.long(), g_post)
        return g_v, g_s, g_params.sum(1)

    @synapse_step_backward_op.register_fake
    def _synapse_step_backward_fake(v, s, pre, targets, p0, p1, p2, p3, p4, p5, area, scalars, inv, pre_nodes, g_s_new, g_current, g_lin, g_const, k, model, targets_are_nodes, has_g_s_new, has_g_current, need):
        g_s = torch.empty_like(s) if model == KINETIC else v.new_empty(0)
        return torch.empty_like(v), g_s, v.new_empty(bin(need).count("1"), pre.shape[0])

    def _setup_context(ctx, inputs, output):
        v, s, pre, targets, p0, p1, p2, p3, p4, p5, area, scalars, inv, nodes, k, model, tan, store = inputs
        ctx.save_for_backward(v, s, pre, targets, p0, p1, p2, p3, p4, p5, area, scalars, inv, nodes)
        ctx.k, ctx.model, ctx.tan = k, model, tan

    def _backward(ctx, g_s_new, g_current, g_lin, g_const):
        v, s, pre, targets, p0, p1, p2, p3, p4, p5, area, scalars, inv, nodes = ctx.saved_tensors
        n_params = 6 if ctx.model == KINETIC else 3
        need = sum(1 << i for i in range(n_params) if ctx.needs_input_grad[4 + i])
        B, T = v.shape[0], pre.shape[0] // ctx.k
        empty = v.new_empty(0)
        has_g_s_new = ctx.model == KINETIC and g_s_new is not None
        has_g_current = g_current is not None and g_current.numel() > 0
        g_lin = g_lin.contiguous() if g_lin is not None else v.new_zeros(B, T)
        g_const = g_const.contiguous() if g_const is not None else v.new_zeros(B, T)
        g_v, g_s, g_params = synapse_step_backward_op(
            v, s, pre, targets, p0, p1, p2, p3, p4, p5, area, scalars, inv, nodes,
            g_s_new.contiguous() if has_g_s_new else empty,
            g_current.contiguous() if has_g_current else empty, g_lin, g_const,
            ctx.k, ctx.model, ctx.tan, has_g_s_new, has_g_current, need,
        )
        rows = [bin(need & ((1 << i) - 1)).count("1") for i in range(6)]
        g_p = [g_params[rows[i]] if need & (1 << i) else None for i in range(6)]
        return (g_v, g_s if ctx.model == KINETIC else None, None, None, *g_p) + (None,) * 8

    synapse_step_op.register_autograd(_backward, setup_context=_setup_context)


def fused_step(states: Dict, delta_t: float, sp: Dict, gp: Dict) -> Optional[tuple]:
    """`Network._step_regular_synapses` with the fused kernels, or `None` if they do not
    apply (then the caller uses the PyTorch code)."""
    if "kernel_model" not in sp or "kernel_scalars" not in gp:
        return None
    v = states["v"]
    if not v.is_cuda or v.dtype not in (torch.float32, torch.float64) or v.dim() == 0:
        return None
    synapse_type = sp["synapse"]
    model = sp["kernel_model"]
    num_targets, k = sp["pre_per_target"].shape  # Static for `torch.compile`.
    num_syn = num_targets * k
    params = [gp["params"][name] for name in sp["kernel_params"]]
    area = gp["area_post"]
    tensors = params + [area]
    if any(not isinstance(p, torch.Tensor) or p.shape != (num_syn,) or p.device != v.device for p in tensors):
        return None
    if area.requires_grad:
        return None
    params = [p.to(v.dtype).contiguous() for p in params]
    params += [params[0]] * (6 - len(params))  # Not used by `SigmoidRateSynapse`.
    batch, n = v.shape[:-1], v.shape[-1]
    name = synapse_type._name
    state = f"{name}_s"
    if model == KINETIC:
        s = states[state]
        if tuple(s.shape) != (*batch, num_syn):
            return None
        s = s.reshape(-1, num_syn).contiguous()
    else:
        s = v.new_empty(0)
    store = gp.get("store_current", True)
    s_new, current, lin, const = synapse_step_op(
        v.reshape(-1, n).contiguous(), s, sp["kernel_pre"], sp["kernel_targets"], *params,
        area.to(v.dtype).contiguous(), gp["kernel_scalars"], sp["kernel_inv"],
        sp["kernel_pre_nodes"], k, model, sp["targets_are_nodes"], store,
    )
    if model == KINETIC:
        states[state] = s_new.reshape(*batch, num_syn)
    if store:
        states[f"i_{name}"] = current.reshape(*batch, num_syn)
    lin = lin.reshape(*batch, lin.shape[-1])
    const = const.reshape(*batch, const.shape[-1])
    if sp["targets_are_nodes"]:
        return states, (lin, -const)
    zeros = torch.zeros_like(v)
    targets = sp["targets"]
    return states, (zeros.index_add(-1, targets, lin), zeros.index_add(-1, targets, -const))
