"""Paper Fig. 4d-f: a biophysical RNN trained on an evidence-integration task.

Port of `nex/fig5_rnn/scripts/evidence_integration.py` and
`nex/fig5_rnn/notebooks/EI_load_results.ipynb` of
https://github.com/mackelab/jaxley_experiments. 20 recurrent units (10 inhibitory)
receive a low-pass filtered noise stimulus with a positive or a negative mean for
500 ms (20,000 steps); two passive readouts have to report its sign in the following
20 ms. Trained: the 20 input weights and the 109 synaptic conductances.

Modes:
  evaluate  Accuracy over 1,000 trials and the psychometric curve (20 stimulus means,
            100 trials each) of a parameter set: `--params paper` (the paper's trained
            network, Zenodo 14166049 `fig5_results.zip`) or `--params <ours.npz>`.
  train     The paper's training run: Adam (lr 0.01) for 2,001 steps of 4 trials, the
            gradient rescaled to norm(g)**0.2 * 0.01 * norm(g_0), three levels of
            checkpointing, starting from the paper's initial network (seed 3) and
            training seed 568. One training iteration is captured in a CUDA graph.

The paper's training script scales the input by 1 / (10 dt) (`--input-scale 4.0`);
its evaluation notebook does not (`--input-scale 1.0`).

`--legacy 1` (default) simulates as the Jaxley version of the paper did (axial
conductances not divided by the capacitance, see `legacy_jaxley.py`; the units'
apical dendrites have 2 uF/cm^2); `--legacy 0` uses Jaxley v0.14 behaviour.
The paper's run (0802-114242) used a Jaxley with branchpoint nodes (as v0.14): with
`--legacy 1`, the loss of the trial that `EI_load_results.ipynb` prints for the trained
network (0.6844109395442985) is reproduced to 1e-12. `--legacy-coupling 1` couples the
branches without branchpoints, as Jaxley did before 2024-08-01 (see `legacy_jaxley.py`).

Usage: python fig4_evidence_integration.py --out <dir> --mode evaluate --params paper
       python fig4_evidence_integration.py --out <dir> --mode train
"""
import argparse
import contextlib
import json
import math
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
from jaxley_pytorch.optimize.transforms import SigmoidTransform
from legacy_jaxley import use_legacy_axial_conductances, use_legacy_branch_coupling
from rnn_utils import edge_cells, gen_stim, gen_stim_batch, initialize_rnn

REFERENCE = "/nrs/scicompsoft/orhane/probes/jaxley_pytorch/data/zenodo_14166049/fig5_results.zip"
TIME_STAMP = "0802-114242"  # The paper's network.
# The paper's configuration (`evint_params_0802-114242.pkl`).
TASK = {
    "dt": 0.025,
    "loss": "CE",
    "response_length": 20,
    "response_time": 500,
    "stim_freq_max": 2500,
    "stim_mean": [-0.005, 0.005],
    "stim_mean_std": 0.0002,
    "stim_std": 0.05,
    "target_pol": 0,
    "total_time": 520,
}
RNN = {
    "RNN_seed": 3,
    "in_conn_prob": 1,
    "init_gain": 5.0 / (100_000 / 2 / np.pi / 10.0 / 1.0),
    "inp_scale": 0.1,
    "k_minus": 1,
    "n_inh": 10,
    "n_out": 2,
    "n_rec": 20,
    "out_conn_prob": 1,
    "out_k_minus": 0.025,
    "out_scale": 1,
    "rec_conn_prob": 0.2,
}
TRAIN = {"batch_size": 4, "checkpoint_levels": 3, "lr": 0.01, "n_epochs": 2000, "train_seed": 568}
N_REC, N_OUT = RNN["n_rec"], RNN["n_out"]


def reference(name):
    sys.path.insert(0, "/nrs/scicompsoft/orhane/probes/jaxley_pytorch/tools")
    from safe_pickle import load_from_zip

    return load_from_zip(REFERENCE, f"results/{name}_{TIME_STAMP}.pkl")


def build():
    net, conn_matrix, input_weights = initialize_rnn(RNN)
    for i in range(N_REC):
        net.cell(i).branch(1).comp(0).record(verbose=False)
    for i in range(N_OUT):
        net.cell(N_REC + i).branch(0).comp(0).record(verbose=False)
    net.IonotropicSynapse.edge("all").make_trainable("IonotropicSynapse_gS", verbose=False)
    return net, conn_matrix, input_weights


def simulate(net, input_weights, gs, stim, input_scale, checkpoint_lengths=None, compile=False):
    """stim: (batch, time). Returns the voltages (batch, recordings, time + 1)."""
    data_stimuli = None
    scaled = stim * input_scale
    for i in range(N_REC):
        data_stimuli = net.cell(i).branch(2).comp(0).data_stimulate(
            scaled[:, None, :] * input_weights[..., i : i + 1, None], data_stimuli=data_stimuli
        )
    return jx.integrate(
        net,
        delta_t=TASK["dt"],
        params=[{"IonotropicSynapse_gS": gs}],
        data_stimuli=data_stimuli,
        checkpoint_lengths=checkpoint_lengths,
        compile=compile,
    )


def readout_logits(v, mask):
    """Mean over all time points of mask * readout voltage (as in the paper)."""
    return torch.mean(mask[:, None, :] * v[:, -N_OUT:, 1:], dim=-1)


def evaluate(args, device):
    net, conn_matrix, input_weights0 = build()
    if args.params == "paper":
        init_conn, init_w, _ = reference("evint_init_params")
        print("initial connectivity matches the paper:", np.allclose(conn_matrix, init_conn, rtol=1e-12, atol=0),
              "max diff", float(np.max(np.abs(conn_matrix - init_conn))))
        print("initial input weights match the paper:",
              np.allclose(input_weights0, init_w[0]["input_weights"], rtol=1e-12, atol=0))
        params = reference("evint_params")[0]
        w = torch.as_tensor(params[0]["input_weights"], device=device)
        gs = torch.as_tensor(params[1]["IonotropicSynapse_gS"], device=device)
    else:
        trained = np.load(args.params)
        w = torch.as_tensor(trained["input_weights"], device=device)
        gs = torch.as_tensor(trained["gS"], device=device)
    to_t = lambda x: torch.as_tensor(x, device=device)
    results = {"params": args.params, "input_scale": args.input_scale, "legacy": bool(args.legacy), "legacy_coupling": bool(args.legacy_coupling)}

    def run(stims, masks):
        with torch.no_grad():
            out = []
            for s, m in zip(torch.split(to_t(stims), args.eval_batch), torch.split(to_t(masks), args.eval_batch)):
                v = simulate(net, w, gs, s, args.input_scale, compile=bool(args.compile))
                out.append((readout_logits(v, m).cpu().numpy(), v[:, -N_OUT:, 1:].cpu().numpy()))
        return np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out])

    # Accuracy under the training conditions.
    np.random.seed(args.eval_seed)
    stims, targets, masks = gen_stim_batch(TASK, args.num_trials)
    t0 = time.time()
    logits, _ = run(stims, masks)
    results["eval_time_s"] = time.time() - t0
    correct = np.argmax(logits, axis=1) == np.argmax(targets, axis=1)
    results["accuracy"] = float(np.mean(correct))
    print(f"accuracy over {args.num_trials} trials: {results['accuracy']:.4f} ({results['eval_time_s']:.1f} s)", flush=True)

    # Psychometric curve: fraction of trials with r1 > r2 in the response window.
    start = int(TASK["response_time"] / TASK["dt"])
    stop = start + int(TASK["response_length"] / TASK["dt"])
    means = np.arange(-0.005, 0.005, 0.0005)
    stims, masks = [], []
    for mean in means:
        task = dict(TASK, stim_mean=[mean, mean])
        for _ in range(args.psycho_trials):
            s, _, m = gen_stim(task)
            stims.append(s)
            masks.append(m)
    _, readouts = run(np.stack(stims), np.stack(masks))
    positive = readouts[:, 0, start:stop].mean(-1) > readouts[:, 1, start:stop].mean(-1)
    frac = positive.reshape(len(means), args.psycho_trials).mean(1)
    results["psychometric"] = {"means": means.tolist(), "fraction_r1_gt_r2": frac.tolist()}
    if args.params == "paper":
        ref_means, ref_frac, _ = reference("psychometric_curve_data_500ms")
        results["psychometric"]["paper_fraction"] = [float(x) for x in ref_frac]
    print("psychometric:", np.round(frac, 2).tolist())

    # Example trials (Fig. 4d) with means +-0.002.
    np.random.seed(3)
    examples = [gen_stim(dict(TASK, stim_mean=[m, m])) for m in [0.002, -0.002]]
    with torch.no_grad():
        ex_v = simulate(
            net, w, gs, to_t(np.stack([e[0] for e in examples])), args.input_scale
        ).cpu().numpy()
    tag = "paper" if args.params == "paper" else "ours"
    with open(os.path.join(args.out, f"evaluate_{tag}_scale{args.input_scale:g}_{args.variant}.json"), "w") as f:
        json.dump(results, f, indent=1)
    np.savez(
        os.path.join(args.out, f"evaluate_{tag}_scale{args.input_scale:g}_{args.variant}.npz"),
        example_stims=np.stack([e[0] for e in examples]),
        example_v=ex_v,
        input_weights=w.cpu().numpy(),
        gS=gs.cpu().numpy(),
        init_input_weights=input_weights0,
        init_gS=net.edges["IonotropicSynapse_gS"].to_numpy(),
        conn_post=edge_cells(net)[1],
        psycho_means=means,
        psycho_frac=frac,
    )
    plot(args, tag)


def plot(args, tag):
    d = np.load(os.path.join(args.out, f"evaluate_{tag}_scale{args.input_scale:g}_{args.variant}.npz"))
    r = json.load(open(os.path.join(args.out, f"evaluate_{tag}_scale{args.input_scale:g}_{args.variant}.json")))
    dt = TASK["dt"]
    fig, ax = plt.subplots(3, 3, figsize=(12, 7))
    for col in range(2):
        t = np.arange(d["example_stims"].shape[1]) * dt
        ax[0, col].plot(t, d["example_stims"][col], c="k", lw=0.5)
        ax[0, col].set_title(f"d  stimulus mean {[0.002, -0.002][col]}")
        v = d["example_v"][col]
        ax[1, col].imshow(v[:N_REC, 1:] > -50.0, aspect="auto", cmap="binary", origin="lower",
                          extent=[0, t[-1], 0, N_REC])
        ax[1, col].set_ylabel("unit")
        ax[2, col].plot(t, v[N_REC, 1:], c="darkorange", label="r1")
        ax[2, col].plot(t, v[N_REC + 1, 1:], c="firebrick", label="r2")
        ax[2, col].axvspan(TASK["response_time"], TASK["response_time"] + TASK["response_length"], color="0.9")
        ax[2, col].set_xlabel("time (ms)")
        ax[2, col].legend(fontsize=7)
    post = d["conn_post"]
    rec = post < N_REC
    ax[0, 2].hist([d["init_input_weights"], d["input_weights"]], bins=20, label=["initial", "trained"])
    ax[0, 2].set_title("e  input weights")
    ax[0, 2].legend(fontsize=7)
    ax[1, 2].hist([d["init_gS"][rec], d["gS"][rec]], bins=20, label=["initial", "trained"])
    ax[1, 2].set_title("e  recurrent gS")
    psy = r["psychometric"]
    ax[2, 2].scatter(psy["means"], psy["fraction_r1_gt_r2"], c="k", s=8, label="ours")
    if "paper_fraction" in psy:
        ax[2, 2].scatter(psy["means"], psy["paper_fraction"], facecolors="none", edgecolors="#08519c", s=20, label="paper")
    ax[2, 2].axhline(0.5, ls=":", c="grey")
    ax[2, 2].axvline(0, ls=":", c="grey")
    ax[2, 2].set_xlabel("stimulus mean")
    ax[2, 2].set_ylabel("fraction r1 > r2")
    ax[2, 2].set_title(f"f  psychometric (accuracy {r['accuracy']:.3f})")
    ax[2, 2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f"fig4_def_{tag}_scale{args.input_scale:g}_{args.variant}.png"), dpi=150)


def train(args, device):
    net, conn_matrix, input_weights0 = build()
    gs0 = net.get_parameters()[0]["IonotropicSynapse_gS"].to(device)
    lowers = {"input_weights": -0.2, "gS": 1e-10}
    uppers = {"input_weights": 0.2, "gS": float(np.max(np.abs(conn_matrix)) * 3)}
    transforms = {k: SigmoidTransform(lowers[k], uppers[k]) for k in lowers}
    assert np.all(np.abs(input_weights0) < uppers["input_weights"]), "initial input weights out of bounds"
    opt = {
        "input_weights": transforms["input_weights"].inverse(torch.as_tensor(input_weights0, device=device)),
        "gS": transforms["gS"].inverse(gs0),
    }
    opt = {k: v.clone().requires_grad_(True) for k, v in opt.items()}
    n_steps = int(TASK["total_time"] / TASK["dt"])
    levels = TRAIN["checkpoint_levels"]
    checkpoints = [int(math.ceil(n_steps ** (1 / levels)))] * levels
    batch = TRAIN["batch_size"]
    static = {
        "stim": torch.zeros(batch, n_steps, device=device),
        "target": torch.zeros(batch, 2, device=device),
        "mask": torch.zeros(batch, n_steps, device=device),
    }

    def loss_fn():
        w = transforms["input_weights"].forward(opt["input_weights"])
        gs = transforms["gS"].forward(opt["gS"])
        v = simulate(net, w, gs, static["stim"], args.input_scale, checkpoints, compile=bool(args.compile))
        logits = readout_logits(v, static["mask"])
        return -torch.mean(torch.sum(static["target"] * torch.log_softmax(logits, dim=-1), dim=-1))

    def load(stim, target, mask):
        for k, x in zip(["stim", "target", "mask"], [stim, target, mask]):
            static[k].copy_(torch.as_tensor(x, device=device))

    # Everything that touches the parameters before graph capture runs on one side
    # stream (autograd's accumulators remember the stream they were created on).
    side = torch.cuda.Stream() if args.graph else None
    if side is not None:
        side.wait_stream(torch.cuda.current_stream())
    # The paper sets the gradient scale from a first batch drawn right after building
    # the network, then reseeds.
    load(*gen_stim_batch(TASK, batch))
    with torch.cuda.stream(side) if side is not None else contextlib.nullcontext():
        grads = torch.autograd.grad(loss_fn(), list(opt.values()))
        desired_norm = 0.01 * torch.sqrt(sum(torch.sum(g**2) for g in grads))
    if side is not None:
        torch.cuda.current_stream().wait_stream(side)
    beta = 0.8
    optimizer = torch.optim.Adam(list(opt.values()), lr=TRAIN["lr"], capturable=bool(args.graph))

    def iteration():
        optimizer.zero_grad(set_to_none=False)
        loss = loss_fn()
        loss.backward()
        with torch.no_grad():
            norm = torch.sqrt(sum(torch.sum(p.grad**2) for p in opt.values()))
            for p in opt.values():
                p.grad.mul_(desired_norm / norm**beta)
        optimizer.step()
        return loss.detach()

    np.random.seed(TRAIN["train_seed"])
    total = TRAIN["n_epochs"] + 1  # The paper's loop makes an extra step at i == 0.
    losses = torch.zeros(total, device=device)
    graph, static_loss = None, None
    t_start = time.time()
    for step in range(total):
        load(*gen_stim_batch(TASK, batch))
        if graph is not None:
            graph.replay()
        elif side is not None and step < 2:
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                static_loss = iteration()
            torch.cuda.current_stream().wait_stream(side)
        elif side is not None:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_loss = iteration()
            graph.replay()
        else:
            static_loss = iteration()
        losses[step] = static_loss
        if step % 50 == 0 or step == total - 1:
            elapsed = time.time() - t_start
            print(f"step {step}: loss {losses[step].item():.4f}, {elapsed / (step + 1):.2f} s/step", flush=True)
        if step % 200 == 0 or step == total - 1:
            np.savez(
                os.path.join(args.out, "trained.npz"),
                input_weights=transforms["input_weights"].forward(opt["input_weights"]).detach().cpu().numpy(),
                gS=transforms["gS"].forward(opt["gS"]).detach().cpu().numpy(),
                losses=losses[: step + 1].cpu().numpy(),
                step=step,
            )
    print(f"training took {time.time() - t_start:.0f} s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=["evaluate", "train"], required=True)
    parser.add_argument("--params", default="paper")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--num-trials", type=int, default=1000)
    parser.add_argument("--psycho-trials", type=int, default=100)
    parser.add_argument("--eval-batch", type=int, default=1000)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--compile", type=int, default=1)
    parser.add_argument("--graph", type=int, default=1)
    parser.add_argument("--legacy", type=int, default=1, help="behave like the paper's Jaxley")
    parser.add_argument("--legacy-coupling", type=int, default=0,
                        help="couple branches without branchpoints, as Jaxley before 2024-08-01")
    parser.add_argument("--total-time", type=float, default=TASK["total_time"], help="trial length (ms), for smoke tests")
    parser.add_argument("--epochs", type=int, default=TRAIN["n_epochs"])
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_default_dtype(torch.float64)  # As in the paper (jax_enable_x64).
    device = torch.device(args.device)
    args.graph = args.graph and device.type == "cuda"
    use_legacy_axial_conductances(bool(args.legacy))
    use_legacy_branch_coupling(bool(args.legacy_coupling))
    args.variant = f"legacy{args.legacy}_coupling{args.legacy_coupling}"
    if args.total_time != TASK["total_time"]:
        TASK["response_time"] = args.total_time - TASK["response_length"]
        TASK["total_time"] = args.total_time
    TRAIN["n_epochs"] = args.epochs
    if args.mode == "evaluate":
        evaluate(args, device)
    else:
        train(args, device)


if __name__ == "__main__":
    main()
