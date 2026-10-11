"""Paper Fig. 2a-c: fit 19 parameters of an L5PC to a synthetic somatic voltage trace.

Port of `nex/fig3_l5pc/notebooks/01_synthetic.ipynb` of
https://github.com/mackelab/jaxley_experiments. A layer 5 pyramidal cell (339 branches
of 4 compartments) with nine channel types in soma, dendrites and axon receives a
somatic step current (3 nA from 10 to 90 ms). The 19 parameters of the ground-truth
model (maximal conductances and calcium-pump constants) are recovered from four summary
statistics of the somatic voltage: its mean and standard deviation in two time windows.

As in the paper: 10 runs of Polyak gradient descent (the gradient is divided by its
norm**0.8 and the learning rate is loss / 3) for 10 steps each, starting from values
drawn uniformly within the bounds (numpy seeds 42-51, so the runs start from the
paper's initial parameters), and a random search over 10,000 parameter sets (numpy
seed 1, the paper's samples). The 10 runs are trained simultaneously as a batch
dimension. With `--reference`, the paper's results (Zenodo 14166049,
`fig3_results.zip`) are compared and drawn in grey.

`--legacy 1` (default) builds the model as the Jaxley version of the paper did (its SWC
reader, and axial conductances not divided by the capacitance, see `legacy_jaxley.py`):
the ground-truth trace then matches the paper's to 0.03 mV, and the runs follow the
paper's runs. `--legacy 0` uses Jaxley v0.14 (the port's) behaviour.

Usage: python fig2_l5pc_synthetic.py --out <dir> [--legacy 1] [--device cuda] [--reference <zip>]
"""
import argparse
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import jaxley_pytorch as jx
from jaxley_pytorch.channels import Leak
from jaxley_pytorch.mech.channels.l5pc import (
    H,
    M,
    SKE2,
    CaHVA,
    CaLVA,
    CaNernstReversal,
    CaPump,
    KPst,
    KTst,
    NaTaT,
    NaTs2T,
    SKv3_1,
)
from jaxley_pytorch.morphology import distance_direct
from jaxley_pytorch.optimize.transforms import SigmoidTransform

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from legacy_jaxley import legacy_distance, read_swc_legacy, use_legacy_axial_conductances

HERE = os.path.dirname(os.path.abspath(__file__))
SWC = os.path.join(HERE, "..", "..", "tests", "swc_files", "morph_l5pc_with_axon.swc")
REFERENCE = "/nrs/scicompsoft/orhane/probes/jaxley_pytorch/data/zenodo_14166049/fig3_results.zip"
DT, T_MAX = 0.025, 100.0
# (group, parameter): (lower bound, upper bound, ground truth), in the paper's order.
PARAMS = {
    ("apical", "apical_NaTs2T_gNaTs2T"): (0.0, 0.04, 0.026145),
    ("apical", "apical_SKv3_1_gSKv3_1"): (0.0, 0.04, 0.004226),
    ("apical", "apical_M_gM"): (0.0, 0.001, 0.000143),
    ("soma", "somatic_NaTs2T_gNaTs2T"): (0.0, 1.0, 0.983955),
    ("soma", "somatic_SKv3_1_gSKv3_1"): (0.25, 1.0, 0.303472),
    ("soma", "somatic_SKE2_gSKE2"): (0.0, 0.1, 0.008407),
    ("soma", "somatic_CaPump_gamma"): (0.0005, 0.01, 0.000609),
    ("soma", "somatic_CaPump_decay"): (20.0, 1000.0, 210.485291),
    ("soma", "somatic_CaHVA_gCaHVA"): (0.0, 0.001, 0.000994),
    ("soma", "somatic_CaLVA_gCaLVA"): (0.0, 0.01, 0.000333),
    ("axon", "axonal_NaTaT_gNaTaT"): (0.0, 4.0, 3.137968),
    ("axon", "axonal_KPst_gKPst"): (0.0, 1.0, 0.973538),
    ("axon", "axonal_KTst_gKTst"): (0.0, 0.1, 0.089259),
    ("axon", "axonal_SKE2_gSKE2"): (0.0, 0.1, 0.007104),
    ("axon", "axonal_SKv3_1_gSKv3_1"): (0.0, 2.0, 1.021945),
    ("axon", "axonal_CaHVA_gCaHVA"): (0.0, 0.001, 0.00099),
    ("axon", "axonal_CaLVA_gCaLVA"): (0.0, 0.01, 0.008752),
    ("axon", "axonal_CaPump_gamma"): (0.0005, 0.05, 0.00291),
    ("axon", "axonal_CaPump_decay"): (20.0, 1000.0, 287.19873),
}
WINDOWS = [slice(200, 2200), slice(2200, 3800)]
Z_SCORE = [8.0, 4.0, 8.0, 4.0]
CHECKPOINT_LENGTHS = [64, 64]  # As in the paper: ceil(sqrt(4000)) twice.


def build_cell(swc, device, reader="legacy"):
    """`reader="legacy"` builds the paper's morphology with the SWC reader of Jaxley
    v0.3.0 (`legacy_jaxley.py`); `"graph"` uses the reader of Jaxley v0.14."""
    if reader == "legacy":
        cell = read_swc_legacy(swc, nseg=4, max_branch_len=300.0, assign_groups=True)
    else:
        cell = jx.read_swc(swc, ncomp=4, max_branch_len=300.0, assign_groups=True)
    apical_inds = np.unique(cell.apical.nodes["global_branch_index"]).tolist()

    cell.apical.set("capacitance", 2.0)
    cell.apical.insert(NaTs2T().change_name("apical_NaTs2T"))
    cell.apical.insert(SKv3_1().change_name("apical_SKv3_1"))
    cell.apical.insert(M().change_name("apical_M"))
    cell.apical.insert(H().change_name("apical_H"))
    # gH grows with the direct distance between the compartment and the first
    # compartment of branch 0 (`cell.branch(b).loc(c).distance(cell.branch(0).loc(0.0))`).
    for b in apical_inds:
        for k, c in enumerate([0.125, 0.375, 0.625, 0.875]):
            if reader == "legacy":
                distance = legacy_distance(cell, (b, k), (0, 0), nseg=4)
            else:
                distance = float(distance_direct(cell.branch(0).loc(0.0), cell.branch(b).loc(c))[0])
            cond = (-0.8696 + 2.087 * np.exp(distance * 0.0031)) * 8e-5
            cell.branch(b).loc(c).set("apical_H_gH", cond)

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

    cell.insert(CaNernstReversal())
    cell.set("CaCon_i", 5e-05)
    cell.set("CaCon_e", 2.0)
    for name, channel in [
        ("axonal_NaTaT", NaTaT),
        ("axonal_KTst", KTst),
        ("axonal_CaPump", CaPump),
        ("axonal_SKE2", SKE2),
        ("axonal_CaHVA", CaHVA),
        ("axonal_KPst", KPst),
        ("axonal_SKv3_1", SKv3_1),
        ("axonal_CaLVA", CaLVA),
    ]:
        cell.axon.insert(channel().change_name(name))

    cell.insert(Leak())
    cell.set("Leak_gLeak", 3e-05)
    cell.set("Leak_eLeak", -75.0)
    cell.set("axial_resistivity", 100.0)
    cell.set("eNa", 50.0)
    cell.set("eK", -85.0)
    cell.set("v", -65.0)
    for (group, key), (_, _, gt) in PARAMS.items():
        getattr(cell, group).set(key, gt)

    soma_branch = int(np.unique(cell.soma.nodes["global_branch_index"])[0])
    amp = torch.tensor(3.0, device=device)
    cell.branch(soma_branch).loc(0.0).stimulate(
        jx.step_current(10.0, 80.0, amp, DT, T_MAX), verbose=False
    )
    cell.branch(soma_branch).loc(0.0).record(verbose=False)
    cell.init_states()
    return cell


def simulate(cell, values, compile, checkpoint_lengths=None):
    """values: (batch, 19). Returns the somatic voltage, (batch, time)."""
    pstate = None
    for i, (group, key) in enumerate(PARAMS):
        pstate = getattr(cell, group).data_set(key, values[:, i : i + 1], pstate)
    v = jx.integrate(
        cell,
        param_state=pstate,
        checkpoint_lengths=checkpoint_lengths,
        compile=compile,
    )
    return v[:, 0]


def summary_stats(v):
    stats = []
    for window in WINDOWS:
        stats += [v[..., window].mean(-1), v[..., window].std(-1, correction=0)]
    return torch.stack(stats, dim=-1)


def load_reference(path):
    if not path or not os.path.exists(path):
        return None
    sys.path.insert(0, "/nrs/scicompsoft/orhane/probes/jaxley_pytorch/tools")
    from safe_pickle import load_from_zip

    def get(name):
        return load_from_zip(path, f"results/01_synthetic/{name}.pkl")

    return {
        "gd_losses": np.array([[float(x) for x in run] for run in get("losses_of_every_starting_point")]),
        "ga_losses": np.array([[float(x) for x in run] for run in get("genetic_algorithm_losses")]),
        "x_o": np.asarray(get("x_o"))[0],
        "gd_time_s": float(get("gradient_descent_total_time")),
        "random_search_time_s": float(get("random_search_total_time")),
        "ga_time_s": float(get("genetic_algorithm_total_time")),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--swc", default=SWC)
    parser.add_argument("--legacy", type=int, default=1, help="behave like the paper's Jaxley")
    parser.add_argument("--reference", default=REFERENCE)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--num-random", type=int, default=10_000)
    parser.add_argument("--random-batch", type=int, default=1000)
    parser.add_argument("--compile", type=int, default=1)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)
    torch.set_default_dtype(torch.float64)  # As in the paper (jax_enable_x64).
    sync = torch.cuda.synchronize if device.type == "cuda" else lambda: None

    use_legacy_axial_conductances(bool(args.legacy))
    cell = build_cell(args.swc, device, "legacy" if args.legacy else "graph")
    print(f"{cell.total_nbranches} branches, {cell.total_nbranches * 4} compartments", flush=True)
    lowers = torch.tensor([b[0] for b in PARAMS.values()], device=device)
    uppers = torch.tensor([b[1] for b in PARAMS.values()], device=device)
    gt = torch.tensor([b[2] for b in PARAMS.values()], device=device)
    z_score = torch.tensor(Z_SCORE, device=device)
    transform = SigmoidTransform(lowers, uppers)

    with torch.no_grad():
        x_o = simulate(cell, gt[None], args.compile)[0]
    ss_o = summary_stats(x_o)

    def loss_fn(values, checkpoint_lengths=None):
        v = simulate(cell, values, args.compile, checkpoint_lengths)
        return torch.sum(torch.abs((summary_stats(v) - ss_o) / z_score), dim=-1), v

    # Polyak gradient descent, all runs at once.
    beta = 0.8
    seeds = np.arange(42, 52)
    init = []
    for seed in seeds:
        np.random.seed(seed)
        init.append([np.random.rand() * (hi - lo) + lo for lo, hi, _ in PARAMS.values()])
    theta = transform.inverse(torch.tensor(init, device=device)).clone().requires_grad_(True)
    best_loss = torch.full((len(seeds),), float("inf"), device=device)
    best_theta = theta.detach().clone()
    gd_losses, gd_times = [], []
    for it in range(args.iterations):
        sync()
        t0 = time.time()
        loss, _ = loss_fn(transform.forward(theta), CHECKPOINT_LENGTHS)
        (grad,) = torch.autograd.grad(loss.sum(), theta)
        with torch.no_grad():
            grad_norm = grad.norm(dim=-1, keepdim=True)
            better = loss < best_loss
            best_loss = torch.where(better, loss, best_loss)
            best_theta[better] = theta[better]
            theta -= (loss / 3.0)[:, None] * grad / grad_norm**beta
        sync()
        gd_times.append(time.time() - t0)
        gd_losses.append(loss.detach().cpu().numpy())
        print(
            f"iteration {it}: losses {np.array2string(gd_losses[-1], precision=3)}, "
            f"{gd_times[-1]:.1f} s",
            flush=True,
        )
    gd_losses = np.stack(gd_losses, axis=1)  # (runs, iterations)
    with torch.no_grad():
        best_params = transform.forward(best_theta)
        _, best_traces = loss_fn(best_params)

    # Random search with the paper's samples.
    np.random.seed(1)
    samples = np.random.rand(args.num_random, len(PARAMS))
    samples = torch.tensor(samples, device=device) * (uppers - lowers) + lowers
    random_losses = []
    sync()
    t0 = time.time()
    with torch.no_grad():
        for chunk in torch.split(samples, args.random_batch):
            random_losses.append(loss_fn(chunk)[0].cpu().numpy())
    sync()
    random_time = time.time() - t0
    random_losses = np.concatenate(random_losses)

    x_o = x_o.cpu().numpy()
    best_traces = best_traces.cpu().numpy()
    best_loss = best_loss.cpu().numpy()
    results = {
        "legacy": bool(args.legacy),
        "num_branches": int(cell.total_nbranches),
        "gd_losses": gd_losses.tolist(),
        "gd_best_loss_per_run": best_loss.tolist(),
        "gd_runs_below_1": int(np.sum(best_loss < 1.0)),
        "gd_time_per_iteration_s": gd_times,
        "gd_total_time_s": float(np.sum(gd_times)),
        "random_search_time_s": random_time,
        "random_search_num_below_0.55": int(np.sum(random_losses < 0.55)),
        "random_search_simulations_per_success": float(
            len(random_losses) / max(np.sum(random_losses < 0.55), 1)
        ),
        "random_search_best_loss": float(random_losses.min()),
        "best_params": {k: float(v) for (_, k), v in zip(PARAMS, best_params[np.argmin(best_loss)])},
    }
    ref = load_reference(args.reference)
    if ref is not None:
        n = min(len(ref["x_o"]), len(x_o))
        results["reference"] = {
            "x_o_max_abs_diff_mV": float(np.max(np.abs(ref["x_o"][:n] - x_o[:n]))),
            "x_o_summary_stats_paper": summary_stats(torch.tensor(ref["x_o"])).tolist(),
            "x_o_summary_stats_ours": ss_o.tolist(),
            "gd_losses_paper": ref["gd_losses"].tolist(),
            "gd_best_loss_per_run_paper": ref["gd_losses"].min(axis=1).tolist(),
            "gd_time_s_paper_cpu": ref["gd_time_s"],
            "random_search_time_s_paper_cpu": ref["random_search_time_s"],
        }
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    np.savez(
        os.path.join(args.out, "results.npz"),
        x_o=x_o,
        best_traces=best_traces,
        gd_losses=gd_losses,
        random_losses=random_losses,
    )

    time_vec = np.arange(len(x_o)) * DT
    fig, ax = plt.subplots(1, 3, figsize=(12, 3), gridspec_kw={"width_ratios": [1, 2, 1.4]})
    cell.vis(ax=ax[0], color="k")
    ax[0].set_aspect("equal")
    ax[0].axis("off")
    ax[0].set_title("a  morphology")
    order = np.argsort(best_loss)
    for i in order[1:][best_loss[order[1:]] < 1.0]:
        ax[1].plot(time_vec, best_traces[i], c="#9ecae1", lw=0.8)
    ax[1].plot(time_vec, best_traces[order[0]], c="#08519c", lw=1.0, label="best fit")
    ax[1].plot(time_vec, x_o, c="k", lw=1.0, label="synthetic data")
    for window in WINDOWS:
        ax[1].axvspan(window.start * DT, window.stop * DT, color="0.92" if window.start == 200 else "0.85", zorder=-1)
    ax[1].set_xlabel("time (ms)")
    ax[1].set_ylabel("voltage (mV)")
    ax[1].legend(loc="lower right", fontsize=7)
    ax[1].set_title("b  synthetic data and gradient descent fits")
    iters = np.arange(1, gd_losses.shape[1] + 1)
    if ref is not None:
        ax[2].plot(iters, ref["gd_losses"].min(axis=0), c="0.6", ls="--", label="GD min (paper)")
        ax[2].plot(iters, ref["ga_losses"].min(axis=0), c="k", label="GA min (paper)")
    for run in gd_losses:
        ax[2].plot(iters, run, c="#9ecae1", lw=0.8)
    ax[2].plot(iters, gd_losses.min(axis=0), c="#08519c", lw=1.5, label="GD min (ours)")
    ax[2].set_yscale("log")
    ax[2].set_xlabel("iteration")
    ax[2].set_ylabel("loss")
    ax[2].legend(fontsize=7)
    ax[2].set_title("c  training")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig2_abc.png"), dpi=150)
    print(json.dumps({k: v for k, v in results.items() if k != "reference"}, indent=1))
    if ref is not None:
        r = results["reference"]
        print("x_o max abs diff vs paper (mV):", r["x_o_max_abs_diff_mV"])
        print("summary stats paper", np.round(r["x_o_summary_stats_paper"], 4), "ours", np.round(r["x_o_summary_stats_ours"], 4))
        print("best loss per run, paper:", np.round(r["gd_best_loss_per_run_paper"], 3))
        print("best loss per run, ours: ", np.round(best_loss, 3))


if __name__ == "__main__":
    main()
