"""Paper Fig. 2d (and Extended Data Fig. 5): fit L5PC models to patch-clamp recordings of
four cells of the Allen Cell Types Database with gradient descent.

Port of `nex/fig3_l5pc/notebooks/03_allen_fits/{build_simulator.py, channels.py,
loss_util.py, posthoc_summary_stats.py, gradient_descent/run_gd.py}` of
https://github.com/mackelab/jaxley_experiments (data: `cell_types/specimen_*` there). Each
reconstructed cell gets the L5PC channels (`allen_channels.py`) and 25 free parameters.
As in the paper: 1,000 gradient-descent runs in parallel for 50 steps on the first
200 ms of the recording, in a standard-normal reparameterization of the uniform prior
(probit), with the gradient divided by its norm**0.99 and a learning rate of
5e-4 * loss. The loss is the soft-DTW divergence (gamma 0.1) between the simulated and
recorded voltage after a moving maximum (window 50, stride 30) and scaling to [0, 1]
with the recording's range, with the paper's time penalty; divided by 0.05. Afterwards,
the best parameters of every run are simulated for 1,100 ms, and the best run is the one
with the smallest weighted error of spike count, time and amplitude of the first two
spikes.

The paper ran these fits with Jaxley v0.4.0 (2025), which divided axial conductances by
the capacitance already, but still read SWC files with the old reader: `--legacy 1`
(default) uses that reader (`legacy_jaxley.py`).

Usage: python fig2_allen.py --out <dir> --specimen 485574832 [--runs 1000] [--steps 50]
"""
import argparse
import json
import os
import sys
import time
from math import log10

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jaxley_pytorch as jx
from allen_channels import H, M, SKE2, CaHVA, CaLVA, CaNernstReversal, CaPump, KPst, KTst, NaTaT, NaTs2T, SKv3_1
from jaxley_pytorch.channels import Leak
from legacy_jaxley import legacy_distance, read_swc_legacy

DATA = os.path.expanduser("~/projects/jaxley_experiments/nex/fig3_l5pc/notebooks/03_allen_fits/cell_types")
DT = 0.025
I_DELAY, I_DUR = 50.0, 1000.0
SPECIMENS = {  # rotation, v_init, gleak, capacitance, eleak (`set_setup`, paper code)
    "488683425": (195, -83.15625, 1e-4, 2.0, -88.0),
    "485574832": (155, -85.6, 1e-4, 6.0, -88.0),
    "480353286": (170, -88.90625, 1e-4, 6.0, -95.0),
    "473601979": (195, -89.06251, 1e-4, 2.5, -95.0),
}
# (view, parameter, log10?), in the paper's order, and bounds (`get_bounds`).
PARAMS = [
    ("cell", "HVA_tau", True, (log10(0.2), log10(5.0))),
    ("cell", "LVA_tau", True, (log10(0.2), log10(5.0))),
    ("cell", "vt", False, (0.0, 10.0)),
    ("cell", "eK", False, (-100.0, -70.0)),
    ("cell", "eNa", False, (40.0, 60.0)),
    ("soma", "somatic_NaTs2T_gNaTs2T", False, (0.0, 6.0)),
    ("soma", "somatic_SKv3_1_gSKv3_1", False, (0.25, 1.0)),
    ("soma", "somatic_SKE2_gSKE2", False, (0.0, 0.1)),
    ("soma", "somatic_CaHVA_gCaHVA", False, (0.0, 0.001)),
    ("soma", "somatic_CaLVA_gCaLVA", False, (0.0, 0.01)),
    ("soma", "somatic_CaPump_gamma", False, (0.0005, 0.05)),
    ("soma", "somatic_CaPump_decay", True, (log10(1), log10(100))),
    ("apical", "apical_NaTs2T_gNaTs2T", False, (0.0, 0.04)),
    ("apical", "apical_SKv3_1_gSKv3_1", False, (0.0, 0.001)),
    ("apical", "apical_M_gM", False, (0.0, 0.1)),
    ("apical", "apical_M_tau", True, (log10(0.2), log10(5.0))),
    ("axon", "axonal_NaTaT_gNaTaT", False, (0.0, 6.0)),
    ("axon", "axonal_KPst_gKPst", False, (0.0, 1.0)),
    ("axon", "axonal_KTst_gKTst", False, (0.0, 0.1)),
    ("axon", "axonal_SKE2_gSKE2", False, (0.0, 0.1)),
    ("axon", "axonal_SKv3_1_gSKv3_1", False, (0.0, 2.0)),
    ("axon", "axonal_CaHVA_gCaHVA", False, (0.0, 0.001)),
    ("axon", "axonal_CaLVA_gCaLVA", False, (0.0, 0.01)),
    ("axon", "axonal_CaPump_gamma", False, (0.0005, 0.05)),
    ("axon", "axonal_CaPump_decay", True, (log10(1), log10(100))),
]


def load_ephys(specimen, t_max):
    """Recording and stimulus amplitude (`get_experimental_data`, paper code)."""
    sys.path.insert(0, "/nrs/scicompsoft/orhane/probes/jaxley_pytorch/tools")
    from safe_pickle import loads

    with open(os.path.join(DATA, f"specimen_{specimen}", "ephys_01.pkl"), "rb") as f:
        ephys = loads(f.read())
    dt_stim = np.mean(np.diff(ephys["time"]))
    step = int(DT / dt_stim / 1000)
    stim = np.asarray(ephys["stimulus"])[::step]
    rec = np.asarray(ephys["response"])[::step] - 14.0  # Junction potential.
    onset = np.where(stim > 0.05)[0][0]
    start = int(onset - 50.0 / 0.025)
    offset = np.where(stim < 0.05)[0]
    offset = offset[offset > 20_000][0]
    end = int(offset + 150.0 / 0.025)
    stim, rec = stim[start:end], rec[start:end]
    cut = int((t_max + 2 * DT) / DT)
    return rec[:cut].astype(np.float64), float(np.max(stim))


def build_cell(specimen, legacy, device):
    rotation, v_init, gleak, capacitance, eleak = SPECIMENS[specimen]
    swc = os.path.join(DATA, f"specimen_{specimen}", "reconstruction.swc")
    if legacy:
        cell = read_swc_legacy(swc, nseg=4, max_branch_len=300.0, assign_groups=True)
    else:
        cell = jx.read_swc(swc, ncomp=4, max_branch_len=300.0, assign_groups=True)
    soma_inds = np.unique(cell.soma.nodes["global_branch_index"]).tolist()
    apical_inds = np.unique(cell.apical.nodes["global_branch_index"]).tolist()
    cell.insert(Leak())
    cell.apical.insert(NaTs2T().change_name("apical_NaTs2T"))
    cell.apical.insert(SKv3_1().change_name("apical_SKv3_1"))
    cell.apical.insert(M().change_name("apical_M"))
    cell.apical.insert(H().change_name("apical_H"))
    for b in apical_inds:
        for c in range(4):
            # `.distance(cell.branch(soma).loc(0.5))`; loc 0.5 is compartment 1 of 4.
            if legacy:
                distance = legacy_distance(cell, (b, c), (soma_inds[0], 1), nseg=4)
            else:
                from jaxley_pytorch.morphology import distance_direct

                distance = float(distance_direct(cell.branch(soma_inds[0]).comp(1), cell.branch(b).comp(c))[0])
            cell.branch(b).comp(c).set("apical_H_gH", (-0.8696 + 2.087 * np.exp(distance * 0.0031)) * 8e-5)

    cell.soma.insert(NaTs2T().change_name("somatic_NaTs2T"))
    cell.soma.insert(SKv3_1().change_name("somatic_SKv3_1"))
    cell.soma.insert(SKE2().change_name("somatic_SKE2"))
    ca_dynamics = CaNernstReversal()
    ca_dynamics.channel_constants["T"] = 307.15
    cell.soma.insert(ca_dynamics)
    cell.soma.insert(CaPump().change_name("somatic_CaPump"))
    cell.soma.insert(CaHVA().change_name("somatic_CaHVA"))
    cell.soma.insert(CaLVA().change_name("somatic_CaLVA"))
    cell.soma.set("CaCon_i", 5e-05)
    cell.soma.set("CaCon_e", 2.0)
    cell.basal.insert(H().change_name("basal_H"))
    cell.basal.set("basal_H_gH", 8e-5)
    cell.axon.insert(NaTaT().change_name("axonal_NaTaT"))
    cell.axon.insert(KPst().change_name("axonal_KPst"))
    cell.axon.insert(KTst().change_name("axonal_KTst"))
    cell.axon.insert(SKE2().change_name("axonal_SKE2"))
    cell.axon.insert(SKv3_1().change_name("axonal_SKv3_1"))
    cell.axon.insert(ca_dynamics)  # As in the paper: the somatic instance (T = 307.15 K).
    cell.set("CaCon_i", 5e-05)
    cell.set("CaCon_e", 2.0)
    cell.axon.insert(CaHVA().change_name("axonal_CaHVA"))
    cell.axon.insert(CaLVA().change_name("axonal_CaLVA"))
    cell.axon.insert(CaPump().change_name("axonal_CaPump"))
    cell.set("axial_resistivity", 100.0)
    cell.set("capacitance", capacitance)
    cell.set("Leak_eLeak", eleak)
    cell.set("Leak_gLeak", gleak)
    return cell, soma_inds[0], v_init


def setup_stimulus(cell, soma_branch, v_init, i_amp, t_max, device):
    cell.delete_stimuli()
    cell.delete_recordings()
    amp = torch.tensor(i_amp, device=device)
    cell.branch(soma_branch).comp(0).stimulate(jx.step_current(I_DELAY, I_DUR, amp, DT, t_max), verbose=False)
    cell.branch(soma_branch).comp(0).record(verbose=False)
    cell.set("v", v_init)
    cell.init_states()


def simulate(cell, theta, t_max, compile, checkpointing=True):
    """theta: (batch, 25) in the prior's units (log10 for time constants)."""
    pstate = None
    for i, (view, key, log, _) in enumerate(PARAMS):
        value = 10 ** theta[:, i : i + 1] if log else theta[:, i : i + 1]
        target = cell if view == "cell" else getattr(cell, view)
        pstate = target.data_set(key, value, pstate)
    steps = len(np.arange(0, t_max + 2 * DT, DT))
    ckpt = [int(np.ceil(steps ** (1 / 3)))] * 3 if checkpointing else None
    return jx.integrate(cell, param_state=pstate, checkpoint_lengths=ckpt, compile=compile)[:, 0]


def window_max(x, window=50, stride=30):
    """Maximum over windows [i - window, i) for i = window, window + stride, ... < T."""
    num = len(np.arange(window, x.shape[-1], stride))
    return x.unfold(-1, window, stride)[..., :num, :].amax(-1)


def soft_dtw(x, y, gamma=0.1, power=1.0, big=1e10):
    """Soft-DTW discrepancy of the rows of x (B, n) and y (B or 1, n), with the paper's
    cost |x_i - y_j|**power + |i + j - (n - 1)| / n, by anti-diagonals."""
    n = x.shape[-1]
    i = torch.arange(n, device=x.device)
    cost = torch.abs(x[..., :, None] - y[..., None, :]) ** power
    cost = cost + torch.abs(i[:, None] + i[None, :] - (n - 1)).to(x.dtype) / n
    b = cost.shape[0]
    big_col = torch.full((b, 1), big, dtype=x.dtype, device=x.device)
    prev2 = torch.cat([torch.zeros_like(big_col), torch.full((b, n), big, dtype=x.dtype, device=x.device)], -1)
    prev1 = torch.full((b, n + 1), big, dtype=x.dtype, device=x.device)
    rows = torch.arange(n + 1, device=x.device)
    for s in range(2, 2 * n + 1):  # D(i, j) with 1-based i, j and i + j = s.
        j = s - rows
        valid = (rows >= 1) & (rows <= n) & (j >= 1) & (j <= n)
        c = cost[:, (rows - 1).clamp(0, n - 1), (j - 1).clamp(0, n - 1)]
        options = torch.stack([torch.cat([big_col, prev2[:, :-1]], -1), torch.cat([big_col, prev1[:, :-1]], -1), prev1], -1)
        best = -gamma * torch.logsumexp(-options / gamma, -1)
        current = torch.where(valid, c + best, big)
        prev2, prev1 = prev1, current
    return prev1[:, n]


def find_spikes(v, thr):
    diff = np.diff(v)
    spike = (diff[:-1] > 1e-5) & (diff[1:] < -1e-5) & (v[1:-1] > thr)
    spike = np.pad(spike, (1, 1), constant_values=False)
    peak = np.pad(np.diff(v, 2) < -1e-3, (2, 0), constant_values=False)
    return spike & peak


def spike_features(v, thr):
    s = np.nonzero(find_spikes(v, thr))[0]
    count = len(s)
    t1, t2 = (s[0] * DT if count > 0 else -100.0), (s[1] * DT if count > 1 else -100.0)
    a1, a2 = (v[s[0]] if count > 0 else -100.0), (v[s[1]] if count > 1 else -100.0)
    return np.asarray([count, t1, t2, a1, a2], dtype=float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--specimen", default="485574832", choices=list(SPECIMENS))
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--t-max", type=float, default=199.5)
    parser.add_argument("--chunk", type=int, default=1000, help="runs per batch")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--legacy", type=int, default=1)
    parser.add_argument("--compile", type=int, default=1)
    parser.add_argument("--graph", type=int, default=1, help="CUDA graph of a gradient step")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_default_dtype(torch.float64)  # As in the paper (jax_enable_x64).
    device = torch.device("cuda")
    normal = torch.distributions.Normal(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))
    lower = torch.tensor([p[3][0] for p in PARAMS], device=device)
    upper = torch.tensor([p[3][1] for p in PARAMS], device=device)
    to_normal = lambda u: normal.icdf((u - lower) / (upper - lower))
    from_normal = lambda z: normal.cdf(z) * (upper - lower) + lower

    x_o, i_amp = load_ephys(args.specimen, args.t_max)
    cell, soma, v_init = build_cell(args.specimen, bool(args.legacy), device)
    setup_stimulus(cell, soma, v_init, i_amp, args.t_max, device)
    print(f"specimen {args.specimen}: {cell.total_nbranches} branches, stimulus {i_amp:.3f} nA", flush=True)

    x_o_t = torch.as_tensor(x_o, device=device)
    x_red = window_max(x_o_t)
    lo, hi = x_red.min(), x_red.max()
    y = ((x_red - lo) / (hi - lo))[None]
    c1 = soft_dtw(y, y)

    def loss_fn(z):
        v = simulate(cell, from_normal(z), args.t_max, bool(args.compile))
        x = (window_max(v[..., : len(x_o)]) - lo) / (hi - lo)
        return (soft_dtw(x, y) - 0.5 * soft_dtw(x, x) - 0.5 * c1) / 0.05

    g = torch.Generator(device=device).manual_seed(args.seed)
    u = lower + (upper - lower) * torch.rand(args.runs, len(PARAMS), generator=g, device=device)
    static = {
        "z": to_normal(u),
        "loss": torch.zeros(args.runs, device=device),
        "best_loss": torch.full((args.runs,), 1e10, device=device),
    }
    static["best_z"] = static["z"].clone()

    def gd_step():
        """One step of all runs (Polyak gradient descent), in place on `static`."""
        losses, grads = [], []
        for zc in torch.split(static["z"], args.chunk):
            zc = zc.detach().requires_grad_(True)
            loss = loss_fn(zc)
            (gc,) = torch.autograd.grad(loss.sum(), zc)
            losses.append(loss.detach())
            grads.append(gc)
        loss, grad = torch.cat(losses), torch.cat(grads)
        better = loss < static["best_loss"]
        static["best_loss"].copy_(torch.where(better, loss, static["best_loss"]))
        static["best_z"].copy_(torch.where(better[:, None], static["z"], static["best_z"]))
        norm = torch.sqrt(torch.sum(grad**2, -1, keepdim=True))
        static["z"].copy_(static["z"] - (loss * args.lr)[:, None] * grad / norm**0.99)
        static["loss"].copy_(loss)

    # With `--graph`, the first step runs normally (on a side stream), one step is
    # captured in a CUDA graph and replayed for all remaining steps.
    graph, side = None, torch.cuda.Stream() if args.graph else None
    history, t_start = [], time.time()
    for step in range(args.steps):
        if graph is not None:
            graph.replay()
        elif side is not None and step == 0:
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                gd_step()
            torch.cuda.current_stream().wait_stream(side)
        elif side is not None:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                gd_step()
            graph.replay()
        else:
            gd_step()
        history.append(static["best_loss"].clone())
        loss = static["loss"]
        print(f"step {step}: loss mean {loss.nanmean().item():.2f}, median {loss.nanmedian().item():.2f}, "
              f"min {loss.min().item():.2f} ({(time.time() - t_start) / (step + 1):.1f} s/step)", flush=True)
    history = [h.cpu().numpy() for h in history]
    best_loss, best_z = static["best_loss"], static["best_z"]
    train_time = time.time() - t_start

    # Post hoc: simulate the best parameters of every run for 1,100 ms.
    t_long = 1100.0
    x_long, _ = load_ephys(args.specimen, t_long)
    setup_stimulus(cell, soma, v_init, i_amp, t_long, device)
    with torch.no_grad():
        traces = torch.cat([simulate(cell, from_normal(zc), t_long, bool(args.compile), checkpointing=False)
                            for zc in torch.split(best_z, args.chunk)]).cpu().numpy()[:, : len(x_long)]
    thr = -30.0 if args.specimen == "473601979" else -20.0
    np.random.seed(0)
    observed = spike_features(x_long + np.random.randn(len(x_long)) * 1e-2, thr)
    features = np.stack([spike_features(t, thr) for t in traces])
    errors = np.abs(features - observed) @ np.asarray([5.0, 2.0, 2.0, 1.0, 1.0])
    best = int(np.nanargmin(errors))
    results = {
        "specimen": args.specimen, "legacy": bool(args.legacy), "runs": args.runs, "steps": args.steps,
        "train_time_s": train_time, "best_losses": np.sort(best_loss.cpu().numpy())[:10].tolist(),
        "best_error": float(errors[best]), "observed_features": observed.tolist(),
        "best_features": features[best].tolist(),
        "best_params": {p[1]: float(v) for p, v in zip(PARAMS, from_normal(best_z[best]).cpu().numpy())},
    }
    with open(os.path.join(args.out, f"fit_{args.specimen}.json"), "w") as f:
        json.dump(results, f, indent=1)
    np.savez(os.path.join(args.out, f"fit_{args.specimen}.npz"), history=np.stack(history), best_trace=traces[best],
             observation=x_long, errors=errors)
    print(json.dumps(results, indent=1))

    t = np.arange(len(x_long)) * DT
    fig, ax = plt.subplots(2, 1, figsize=(8, 4), gridspec_kw={"height_ratios": [2, 1]})
    ax[0].plot(t, x_long, c="k", lw=0.6, label="recording")
    ax[0].plot(t, traces[best], c="#08519c", lw=0.6, label="gradient descent fit")
    ax[0].set_ylabel("voltage (mV)")
    ax[0].legend(fontsize=7)
    ax[0].set_title(f"d  Allen cell {args.specimen}: best of {args.runs} runs (error {errors[best]:.1f})")
    hist = np.stack(history)
    ax[1].plot(np.min(hist, axis=1), c="#08519c", label="min over runs")
    ax[1].plot(np.median(hist, axis=1), c="#9ecae1", label="median over runs")
    ax[1].set_xlabel("iteration")
    ax[1].set_ylabel("best loss")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f"fig2_d_{args.specimen}.png"), dpi=150)


if __name__ == "__main__":
    main()
