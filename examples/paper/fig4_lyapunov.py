"""Paper Fig. 4b-c: autonomous dynamics of an untrained biophysical RNN, and its
maximal Lyapunov exponent as a function of the synaptic gain.

Port of `nex/fig5_rnn/notebooks/plot_initial_dynamics_exponents.ipynb` and
`nex/fig5_rnn/scripts/Lyaponuv.py` of https://github.com/mackelab/jaxley_experiments.
The network has 50 recurrent units (10 inhibitory, connection probability 0.2) and two
passive readouts, and receives no input.

As in the paper, the maximal Lyapunov exponent is estimated by evolving a tangent
vector with Jacobian-vector products of one solver step (`torch.func.jvp` through the
step function) and renormalizing it after every step: 80,000 steps (2 s) of transient,
80,000 more for the tangent to align, then the average log growth over 240,000 steps
(6 s). The map acts on the dynamic states (`build_dynamic_state_utils`).

`--map paper` (default) is the map of the paper's `Lyaponuv.py`: the calcium reversal
potential eCa is not part of the state but is reset to the module's stored initial
value (0 mV) in every step, so the calcium current that drives the calcium pump is
computed with it. `--map true` is the network's dynamics (eCa follows CaCon_i); its
exponents are smaller (positive above the same transition gain). All gains of one
network seed run as a batch (the gain only scales the synaptic conductances). Blocks of
`--steps-per-graph` steps are captured in a CUDA graph and replayed.

`--legacy 1` (default) simulates as the Jaxley of the paper's computation (by
2024-07-30) did: axial conductances not divided by the capacitance, and the branches of
every unit coupled without a branchpoint node (see `legacy_jaxley.py`). One step of
`--map paper` then equals one step of the paper's code run on that Jaxley (commit
760e770) to round-off. The exponents stored in the paper's results
(`DMS_Lyapunov7.pkl`, plotted with `--reference`) are not reproduced by the paper's
code either (above the transition they are larger), so they came from a configuration
that is not known.

Usage:
  python fig4_lyapunov.py --out <dir> --mode exponents --seed 1 [--map paper|true]   (Fig. 4c; seeds 0-2)
  python fig4_lyapunov.py --out <dir> --mode traces                (Fig. 4b)
  python fig4_lyapunov.py --out <dir> --mode plot [--reference <zip>]
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jaxley_pytorch as jx
from jaxley_pytorch.integrate import build_init_and_step_fn
from jaxley_pytorch.utils.dynamics import build_dynamic_state_utils
from legacy_jaxley import use_legacy_axial_conductances, use_legacy_branch_coupling
from rnn_utils import initialize_rnn

REFERENCE = "/nrs/scicompsoft/orhane/probes/jaxley_pytorch/data/zenodo_14166049/fig5_results.zip"
DT = 0.025
# `plot_initial_dynamics_exponents.ipynb`, cell 2.
RNN_PARAMS = {
    "RNN_seed": 4,
    "n_rec": 50,
    "n_inh": 10,
    "n_out": 2,
    "out_scale": 0.1,
    "inp_scale": 1,
    "in_conn_prob": 0.1,
    "out_conn_prob": 1,
    "rec_conn_prob": 0.2,
    "k_minus": 1,
    "out_k_minus": 0.1,
}
GAINS = np.arange(0, 0.003 + 0.00025, 0.00025)  # Fig. 4c.
TRACE_GAINS = [0.00075, 0.00125, 0.00275]  # Fig. 4b: stable, regular, chaotic.


def build(seed, gains, device):
    """The network of `seed` with unit gain, and batched parameters for `gains`."""
    net, _, _ = initialize_rnn(dict(RNN_PARAMS, RNN_seed=seed, init_gain=1.0))
    net.IonotropicSynapse.edge("all").make_trainable("IonotropicSynapse_gS", verbose=False)
    unit_gs = net.get_parameters()[0]["IonotropicSynapse_gS"].to(device)
    gains = torch.as_tensor(gains, dtype=unit_gs.dtype, device=device)
    return net, [{"IonotropicSynapse_gS": gains[:, None] * unit_gs}]


def exponents(args, device):
    net, params = build(args.seed, GAINS, device)
    init_fn, step_fn = build_init_and_step_fn(net)
    states, all_params = init_fn(params, delta_t=DT)
    # The state of the map: the dynamic states (`build_dynamic_state_utils`: without
    # currents, branchpoint entries and NaN padding). CaCon_e is constant. With
    # `--map paper`, eCa is not part of the state either but, as in the paper's
    # `Lyaponuv.py`, is taken from the module's stored initial value (0 mV) in every
    # step; the currents (which drive the calcium pump) are recomputed from it.
    remove_obs, add_obs, _, _ = build_dynamic_state_utils(net)
    dyn = remove_obs(states)
    fixed = {"CaCon_e": dyn["CaCon_e"]}
    if args.map == "paper":
        fixed["eCa"] = torch.zeros_like(dyn["eCa"])
    keys = [k for k in dyn if k not in fixed]
    print("tangent states:", keys, flush=True)

    def advance(d):
        full = add_obs({**d, **fixed}, all_params, DT)
        new = remove_obs(step_fn(full, all_params, {}, {}, delta_t=DT))
        return {k: new[k] for k in keys}

    def jvp_step(d, q):
        return torch.func.jvp(advance, (d,), (q,))

    def norm(q):
        return torch.sqrt(sum(torch.sum(v**2, dim=-1) for v in q.values()))

    generator = torch.Generator(device=device).manual_seed(args.seed)
    q = {k: torch.randn(dyn[k].shape, generator=generator, device=device, dtype=dyn[k].dtype) for k in keys}
    r = norm(q)
    q = {k: v / r[:, None] for k, v in q.items()}

    static_s = {k: dyn[k].clone() for k in keys}
    static_q = {k: v.clone() for k, v in q.items()}
    log_growth = torch.zeros(len(GAINS), dtype=states["v"].dtype, device=device)
    K = args.steps_per_graph

    def forward_block():
        s = static_s
        for _ in range(K):
            s = advance(s)
        for k in static_s:
            static_s[k].copy_(s[k])

    def tangent_block():
        s, qq = static_s, static_q
        acc = torch.zeros_like(log_growth)
        for _ in range(K):
            s, qq = jvp_step(s, qq)
            r = norm(qq)
            qq = {k: v / r[:, None] for k, v in qq.items()}
            acc = acc + torch.log(r)
        for k in static_s:
            static_s[k].copy_(s[k])
        for k in static_q:
            static_q[k].copy_(qq[k])
        log_growth.add_(acc)

    def capture(block):
        if device.type != "cuda" or not args.graph:
            return block
        # Warm-up (kernel compilation) on a side stream with copies of the state, so
        # that the dynamics are not advanced.
        saved = ({k: v.clone() for k, v in static_s.items()}, {k: v.clone() for k, v in static_q.items()}, log_growth.clone())
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            block()
        torch.cuda.current_stream().wait_stream(stream)
        for d, src in zip((static_s, static_q), saved[:2]):
            for k in d:
                d[k].copy_(src[k])
        log_growth.copy_(saved[2])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            block()
        return graph.replay

    run_forward = capture(forward_block)
    run_tangent = capture(tangent_block)
    num_transient = args.transient // K
    num_steps = args.steps // K
    timings = {}
    t0 = time.time()
    for _ in range(num_transient):
        run_forward()
    torch.cuda.synchronize() if device.type == "cuda" else None
    timings["transient_s"] = time.time() - t0
    t0 = time.time()
    for _ in range(num_transient):
        run_tangent()
    log_growth.zero_()
    torch.cuda.synchronize() if device.type == "cuda" else None
    timings["align_s"] = time.time() - t0
    t0 = time.time()
    running = []
    for i in range(num_steps):
        run_tangent()
        if (i + 1) % max(num_steps // 20, 1) == 0:
            running.append((log_growth / ((i + 1) * K * DT)).cpu().numpy())
            print(f"{(i + 1) * K} steps: lambda {np.array2string(running[-1], precision=4)}", flush=True)
    torch.cuda.synchronize() if device.type == "cuda" else None
    timings["exponent_s"] = time.time() - t0
    lam = (log_growth / (num_steps * K * DT)).cpu().numpy()  # per ms
    result = {
        "seed": args.seed,
        "legacy": bool(args.legacy),
        "map": args.map,
        "gains": GAINS.tolist(),
        "lyapunov_per_ms": lam.tolist(),
        "running": np.stack(running).tolist(),
        "num_transient_steps": num_transient * K,
        "num_steps": num_steps * K,
        "timings": timings,
        "v_finite": bool(torch.isfinite(static_s["v"]).all()),
    }
    with open(os.path.join(args.out, f"lyapunov_{args.map}_seed{args.seed}.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != "running"}, indent=1))


def traces(args, device):
    net, params = build(RNN_PARAMS["RNN_seed"], TRACE_GAINS, device)
    for i in range(RNN_PARAMS["n_rec"]):
        net.cell(i).branch(1).comp(0).record(verbose=False)
    t_max, transient = 1400.0, 1000.0
    with torch.no_grad():
        v = jx.integrate(net, params=params, t_max=t_max - DT, delta_t=DT, compile=bool(args.compile))
    v = v[..., int(transient / DT) :].cpu().numpy()
    np.save(os.path.join(args.out, "traces.npy"), v)
    print("traces", v.shape)


def plot(args):
    fig, ax = plt.subplots(1, 4, figsize=(12, 2.6))
    tr = np.load(os.path.join(args.out, "traces.npy"))
    time_vec = np.arange(tr.shape[-1]) * DT
    for i, title in enumerate(["stable", "regular", "chaotic"]):
        ax[i].plot(time_vec, tr[i].T, lw=0.4)
        ax[i].set_ylim(-100, 60)
        ax[i].set_title(f"b  gain {TRACE_GAINS[i]}: {title}")
        ax[i].set_xlabel("time (ms)")
    ax[0].set_ylabel("voltage (mV)")
    for map_, ls in [("paper", "-"), ("true", ":")]:
        for seed in range(3):
            path = os.path.join(args.out, f"lyapunov_{map_}_seed{seed}.json")
            if os.path.exists(path):
                r = json.load(open(path))
                ax[3].plot(r["gains"], r["lyapunov_per_ms"], ls, c="k" if seed == 1 else "0.7",
                           label=f"ours ({map_} map), seed {seed}" if seed == 1 else None)
    if args.reference and os.path.exists(args.reference):
        sys.path.insert(0, "/nrs/scicompsoft/orhane/probes/jaxley_pytorch/tools")
        from safe_pickle import load_from_zip

        ref = load_from_zip(args.reference, "results/DMS_Lyapunov7.pkl")
        g, l = np.asarray(ref["gains"]), np.asarray(ref["Lyapunov"])
        n = len(g) // 3
        ax[3].plot(g[n : 2 * n], l[n : 2 * n], c="#08519c", ls="--", label="paper, seed 1")
    ax[3].axhline(0, ls="--", c="grey", lw=0.5)
    ax[3].set_xlim(0, 0.002)
    ax[3].set_ylim(-0.004, 0.05)
    ax[3].set_xlabel("gain")
    ax[3].set_ylabel("lambda (1/ms)")
    ax[3].legend(fontsize=7)
    ax[3].set_title("c  maximal Lyapunov exponent")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig4_bc.png"), dpi=150)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=["exponents", "traces", "plot"], required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--transient", type=int, default=80_000)
    parser.add_argument("--steps", type=int, default=240_000)
    parser.add_argument("--steps-per-graph", type=int, default=200)
    parser.add_argument("--graph", type=int, default=1)
    parser.add_argument("--compile", type=int, default=1)
    parser.add_argument("--reference", default=REFERENCE)
    parser.add_argument("--legacy", type=int, default=1, help="behave like the paper's Jaxley")
    parser.add_argument("--map", choices=["paper", "true"], default="paper",
                        help="paper: eCa reset every step, as in the paper's code; true: the network's dynamics")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_default_dtype(torch.float64)  # As in the paper (jax_enable_x64).
    device = torch.device(args.device)
    use_legacy_axial_conductances(bool(args.legacy))
    use_legacy_branch_coupling(bool(args.legacy))
    if args.mode == "exponents":
        exponents(args, device)
    elif args.mode == "traces":
        traces(args, device)
    else:
        plot(args)


if __name__ == "__main__":
    main()
