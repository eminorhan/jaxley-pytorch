"""Paper Fig. 4g: a biophysical RNN trained on a delayed-match-to-sample task.

Port of `nex/fig5_rnn/scripts/{DMS_run_training.py, DMS_utils.py, parameterisation.py}`
and `nex/fig5_rnn/notebooks/DMS_load_results.ipynb` of
https://github.com/mackelab/jaxley_experiments. 50 recurrent units (10 inhibitory,
connection probability 0.05, 10% of them receive each of the two inputs) see two 50 ms
stimuli (A or B) separated by a delay, and two passive readouts report "match" (AA,
BB) or "non-match" (AB, BA) in the following 50 ms. Trained: input weights (in [0, 4]),
synaptic conductances (softplus, >= 0) and synaptic time constants (k_minus in
[0.05, 2]).

Modes:
  evaluate  The paper's trained network (Zenodo 14166049, `fig5_results.zip`,
            `DMS_18000_22000_2024_07_19_T_07_28_01`) or ours (`--params <npz>`):
            accuracy at the final delays (450-550 ms) and the four patterns with a
            500 ms delay (raster and readouts, Fig. 4g).
  train     The paper's training: batches of 64 trials, Adam on gradients clipped to
            global norm 1, learning rate 0.001 decaying by 0.995 per step to 0.0001,
            two levels of checkpointing, and a curriculum that lengthens the delay by
            100 ms (from 50-150 ms to 450-550 ms) whenever a batch reaches loss < 0.65
            and accuracy > 0.95; 1,000 steps.

`--legacy 1` (default) simulates as the Jaxley version of the paper did (axial
conductances not divided by the capacitance, see `legacy_jaxley.py`).
`--legacy-coupling 1` (default) also couples the branches of every unit without a
branchpoint node, as Jaxley did before 2024-08-01 (see `legacy_jaxley.py`), when the
paper's run in the Zenodo record (2024-07-19) was trained. (The paper's figure notebook,
`DMS_load_results.ipynb`, shows a later network, trained 2024-08-11 with branchpoints,
which is not in the record.) The record's trained network solves the task with either
coupling.

Usage: python fig4_dms.py --out <dir> --mode evaluate [--params paper]
       python fig4_dms.py --out <dir> --mode train
"""
import argparse
import copy
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
from rnn_utils import G_SCALING, initialize_rnn

REFERENCE = "/nrs/scicompsoft/orhane/probes/jaxley_pytorch/data/zenodo_14166049/fig5_results.zip"
NAME = "DMS_18000_22000_2024_07_19_T_07_28_01"
DT = 0.025
RNN = {
    "RNN_seed": 3,
    "seed": 3,
    "n_rec": 50,
    "n_inh": 10,
    "n_out": 2,
    "init_gain": 5 / G_SCALING,
    "out_scale": 0.2,
    "inp_scale": 1,
    "in_conn_prob": 0.1,
    "out_conn_prob": 1,
    "rec_conn_prob": 0.05,
    "k_minus": 1,
    "out_k_minus": 0.1,
}
# Task, in time steps (`DMS_..._config.p`).
TASK = {
    "stim_onset": [800],
    "stim_len": 2000,
    "delay": [2000, 6000],  # Start of the curriculum: 50-150 ms.
    "response_onset": 0,
    "response": 2000,
    "stim_amp": 1,
    "stim_noise_sd": 0.01,
    "batch_size": 64,
}
TRAIN = {"lr": 0.001, "lr_end": 0.0001, "delay_step": 4000, "max_epochs": 1000,
         "loss_threshold": 0.65, "acc_threshold": 0.95, "checkpoint_levels": 2}
N_REC, N_OUT = RNN["n_rec"], RNN["n_out"]


def reference(suffix):
    sys.path.insert(0, "/nrs/scicompsoft/orhane/probes/jaxley_pytorch/tools")
    from safe_pickle import load_from_zip

    return load_from_zip(REFERENCE, f"results/{NAME}_{suffix}.p")


class DMS:
    """Port of the paper's task generator (same random draws)."""

    def __init__(self, params):
        self.params = copy.deepcopy(params)

    def sim_time(self):
        p = self.params
        return p["stim_len"] * 2 + p["delay"][-1] + p["stim_onset"][-1] + p["response_onset"] + p["response"]

    def gen_batch(self):
        p, n = self.params, self.params["batch_size"]
        ramp_up = np.asarray([p["stim_onset"][0]] * n)
        delay = np.random.uniform(low=p["delay"][0], high=p["delay"][1], size=(n,)).astype(int)
        sim_time = self.sim_time()
        stim1_end = ramp_up + p["stim_len"]
        delay_end = stim1_end + delay
        stim2_end = delay_end + p["stim_len"]
        response_onset = stim2_end + p["response_onset"]
        response_end = response_onset + p["response"]
        stim = np.zeros((n, sim_time, 2))
        target = np.zeros((n, sim_time, 2))
        mask = np.zeros((n, sim_time, 2))
        stim1 = np.random.randint(0, 2, n)
        stim2 = np.random.randint(0, 2, n)
        match = (stim1 == stim2).astype(int)
        for i in range(n):
            stim[i, ramp_up[i] : stim1_end[i], stim1[i]] += p["stim_amp"] + np.random.randn(p["stim_len"]) * p["stim_noise_sd"]
            stim[i, delay_end[i] : stim2_end[i], stim2[i]] += p["stim_amp"] + np.random.randn(p["stim_len"]) * p["stim_noise_sd"]
            target[i, response_onset[i] :, match[i]] = 1
            mask[i, response_onset[i] : response_end[i]] = 1
        return stim, target, mask

    def gen(self, delay):
        """The four patterns AA, AB, BA, BB with a fixed delay."""
        p = self.params
        sim_time = p["stim_len"] * 2 + delay + p["stim_onset"][0] + p["response_onset"] + p["response"]
        stim = np.zeros((4, sim_time, 2))
        target = np.zeros((4, sim_time, 2))
        mask = np.zeros((4, sim_time, 2))
        s1_end = p["stim_onset"][0] + p["stim_len"]
        d_end = s1_end + delay
        s2_end = d_end + p["stim_len"]
        for i, (a, b) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            stim[i, p["stim_onset"][0] : s1_end, a] += p["stim_amp"] + np.random.randn(p["stim_len"]) * p["stim_noise_sd"]
            stim[i, d_end:s2_end, b] += p["stim_amp"] + np.random.randn(p["stim_len"]) * p["stim_noise_sd"]
            target[i, s2_end:, int(a == b)] = 1
            mask[i, s2_end : s2_end + p["response"]] = 1
        return stim, target, mask


def softplus(x, b):
    return torch.nn.functional.softplus(x * b) / b


def inv_softplus(y, b):
    return torch.log(torch.expm1(b * y)) / b


class Model:
    def __init__(self, device, compile):
        np.random.seed(RNN["seed"])
        self.net, self.conn_matrix, self.input_weights0 = initialize_rnn(RNN)
        self.net.IonotropicSynapse.edge("all").make_trainable("IonotropicSynapse_gS", verbose=False)
        self.net.IonotropicSynapse.edge("all").make_trainable("IonotropicSynapse_k_minus", verbose=False)
        params = self.net.get_parameters()
        self.gs0 = params[0]["IonotropicSynapse_gS"].to(device)
        self.k0 = params[1]["IonotropicSynapse_k_minus"].to(device)
        for i in range(N_REC):
            self.net.cell(i).branch(1).comp(0).record(verbose=False)
        for i in range(N_OUT):
            self.net.cell(N_REC + i).branch(0).comp(0).record(verbose=False)
        nonzero = self.conn_matrix[self.conn_matrix != 0]
        self.b = 1.0 / np.mean(np.abs(nonzero))
        self.w_tf = SigmoidTransform(0.0, 4.0)
        self.k_tf = SigmoidTransform(0.05, 2.0)
        self.device = device
        self.compile = compile

    def to_opt(self, w, gs, k):
        return {"w": self.w_tf.inverse(w), "gS": inv_softplus(gs, self.b), "k": self.k_tf.inverse(k)}

    def from_opt(self, opt):
        return self.w_tf.forward(opt["w"]), softplus(opt["gS"], self.b), self.k_tf.forward(opt["k"])

    def simulate(self, w, gs, k, stim, checkpoint_levels=None):
        """stim: (batch, time, 2). Each unit gets <stim, w_i> in all its compartments."""
        current = torch.einsum("btc,nc->bnt", stim, w)
        data_stimuli = None
        for i in range(N_REC):
            data_stimuli = self.net.cell(i).data_stimulate(current[:, i : i + 1, :], data_stimuli=data_stimuli)
        ckpt = None
        if checkpoint_levels:
            n = stim.shape[1]
            ckpt = [int(math.ceil(n ** (1 / checkpoint_levels)))] * checkpoint_levels
        return jx.integrate(
            self.net,
            delta_t=DT,
            params=[{"IonotropicSynapse_gS": gs}, {"IonotropicSynapse_k_minus": k}],
            data_stimuli=data_stimuli,
            checkpoint_lengths=ckpt,
            compile=self.compile,
        )


def readout_probs(v):
    """Softmax over the two readouts at every time step, clipped as in the paper."""
    pred = v[:, -N_OUT:, :].transpose(1, 2)  # (batch, time, 2)
    return torch.clamp(torch.softmax(pred, dim=-1), 1e-8, 1 - 1e-8)


def loss_and_accuracy(v, target, mask):
    p = readout_probs(v)
    ce = -torch.sum(mask * (target * torch.log(p) + (1 - target) * torch.log(1 - p)), dim=(1, 2)) / mask.sum(dim=(1, 2))
    votes1 = torch.sum(mask[..., 0] * (p[..., 1] > p[..., 0]), dim=1)
    votes0 = torch.sum(mask[..., 0] * (p[..., 1] < p[..., 0]), dim=1)
    label = torch.argmax(torch.mean(target, dim=1), dim=-1)
    acc = ((votes1 > votes0).long() == label).double()
    return ce.mean(), acc.mean()


def evaluate(args, device):
    model = Model(device, bool(args.compile))
    to_t = lambda x: torch.as_tensor(x, device=device)
    results = {"params": args.params, "legacy": bool(args.legacy), "legacy_coupling": bool(args.legacy_coupling)}
    if args.params == "paper":
        init = {k_: v_ for d in reference("initparams") for k_, v_ in d.items()}
        results["initial_connectivity_matches_paper"] = bool(np.array_equal(init["conn_matrix"], model.conn_matrix))
        results["initial_input_weights_match_paper"] = bool(np.allclose(init["input_weights"], model.input_weights0))
        results["initial_gS_match_paper"] = bool(np.allclose(init["IonotropicSynapse_gS"], model.gs0.cpu().numpy()))
        results["initial_k_minus_match_paper"] = bool(np.allclose(init["IonotropicSynapse_k_minus"], model.k0.cpu().numpy()))
        trained = {k: v for p in reference("params") for k, v in p.items()}
        w = to_t(trained["input_weights"])
        gs, k = to_t(trained["IonotropicSynapse_gS"]), to_t(trained["IonotropicSynapse_k_minus"])
    else:
        d = np.load(args.params)
        w, gs, k = to_t(d["input_weights"]), to_t(d["gS"]), to_t(d["k_minus"])
    print({k_: v for k_, v in results.items()}, flush=True)

    task = dict(TASK, delay=[18000, 22000])
    dms = DMS(task)
    np.random.seed(args.eval_seed)
    accs, losses = [], []
    t0 = time.time()
    with torch.no_grad():
        for _ in range(args.eval_batches):
            stim, target, mask = dms.gen_batch()
            v = model.simulate(w, gs, k, to_t(stim[:, 1:]))
            loss, acc = loss_and_accuracy(v, to_t(target), to_t(mask))
            accs.append(acc.item())
            losses.append(loss.item())
    results.update(accuracy=float(np.mean(accs)), loss=float(np.mean(losses)), num_trials=args.eval_batches * task["batch_size"],
                   eval_time_s=time.time() - t0)
    print(f"accuracy {results['accuracy']:.4f}, loss {results['loss']:.4f} over {results['num_trials']} trials", flush=True)

    stim, target, mask = dms.gen(20000)
    with torch.no_grad():
        v = model.simulate(w, gs, k, to_t(stim[:, 1:])).cpu().numpy()
        _, acc4 = loss_and_accuracy(torch.as_tensor(v), torch.as_tensor(target), torch.as_tensor(mask))
    results["four_pattern_accuracy"] = float(acc4)
    tag = "paper" if args.params == "paper" else "ours"
    with open(os.path.join(args.out, f"evaluate_{tag}_{args.variant}.json"), "w") as f:
        json.dump(results, f, indent=1)
    np.savez(os.path.join(args.out, f"evaluate_{tag}_{args.variant}.npz"), v=v, stim=stim, target=target, mask=mask)

    t = (np.arange(v.shape[-1]) - TASK["stim_onset"][0]) * DT
    fig, ax = plt.subplots(3, 4, figsize=(12, 6), sharex=True)
    for i, name in enumerate(["AA", "AB", "BA", "BB"]):
        ax[0, i].plot(t[1:], stim[i, 1:, 0], c="firebrick", label="A")
        ax[0, i].plot(t[1:], stim[i, 1:, 1], c="darkorange", label="B")
        ax[0, i].set_title(f"g  {name} ({'match' if name[0] == name[1] else 'non-match'})")
        ax[1, i].imshow(v[i, :N_REC] > -50.0, aspect="auto", cmap="binary", origin="lower", extent=[t[0], t[-1], 0, N_REC])
        p = torch.softmax(torch.as_tensor(v[i, -N_OUT:]).T, dim=-1).numpy()
        ax[2, i].plot(t, p[:, 1], c="k", label="p(match)")
        ax[2, i].axvspan(t[np.argmax(mask[i, :, 0])], t[np.argmax(mask[i, :, 0])] + TASK["response"] * DT, color="0.9")
        ax[2, i].set_ylim(-0.05, 1.05)
        ax[2, i].set_xlabel("time (ms)")
    ax[0, 0].legend(fontsize=7)
    ax[1, 0].set_ylabel("unit")
    ax[2, 0].set_ylabel("p(match)")
    fig.suptitle(f"accuracy {results['accuracy']:.3f} ({results['num_trials']} trials, delay 450-550 ms)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f"fig4_g_{tag}_{args.variant}.png"), dpi=150)


def train(args, device):
    model = Model(device, bool(args.compile))
    opt = model.to_opt(torch.as_tensor(model.input_weights0, device=device), model.gs0, model.k0)
    opt = {k: v.clone().requires_grad_(True) for k, v in opt.items()}
    leaves = list(opt.values())
    optimizer = torch.optim.Adam(leaves, lr=TRAIN["lr"], capturable=bool(args.graph))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: max(0.995**s, TRAIN["lr_end"] / TRAIN["lr"]))
    dms = DMS(TASK)
    log = {"loss": [], "accuracy": [], "delay_max": []}
    graphs = {}

    def load(static, stim, target, mask):
        for k_, x in zip(["stim", "target", "mask"], [stim[:, 1:], target, mask]):
            static[k_].copy_(torch.as_tensor(x, device=device))

    def make_static(sim_time):
        b = TASK["batch_size"]
        return {"stim": torch.zeros(b, sim_time - 1, 2, device=device), "target": torch.zeros(b, sim_time, 2, device=device),
                "mask": torch.zeros(b, sim_time, 2, device=device)}

    def compute(static):
        """Loss, accuracy and (clipped) gradients; the step is applied separately because
        the paper skips it whenever the curriculum advances."""
        for x in leaves:
            if x.grad is not None:
                x.grad.zero_()
        w, gs, k = model.from_opt(opt)
        v = model.simulate(w, gs, k, static["stim"], TRAIN["checkpoint_levels"])
        loss, acc = loss_and_accuracy(v, static["target"], static["mask"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(leaves, 1.0, foreach=True)
        return loss.detach(), acc.detach()

    # One side stream for all warm-ups: autograd's accumulators remember the stream on
    # which they were created, and every later capture must agree with it.
    side = torch.cuda.Stream() if args.graph else None
    t_start = time.time()
    for step in range(TRAIN["max_epochs"]):
        sim_time = dms.sim_time()
        stim, target, mask = dms.gen_batch()
        if sim_time not in graphs:
            graphs.clear()  # The curriculum never returns to a shorter trial.
            torch.cuda.empty_cache() if device.type == "cuda" else None
            static = make_static(sim_time)
            load(static, stim, target, mask)
            if args.graph:
                for x in leaves:
                    x.grad = torch.zeros_like(x)
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    out = compute(static)  # Warm-up; gradients are recomputed below.
                torch.cuda.current_stream().wait_stream(side)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out = compute(static)
                graphs[sim_time] = (static, graph, out)
            else:
                graphs[sim_time] = (static, None, None)
        static, graph, out = graphs[sim_time]
        load(static, stim, target, mask)
        if graph is not None:
            graph.replay()
            loss, acc = out
        else:
            loss, acc = compute(static)
        l, a = loss.item(), acc.item()
        log["loss"].append(l)
        log["accuracy"].append(a)
        log["delay_max"].append(dms.params["delay"][1])
        if step % 10 == 0:
            print(f"step {step}: loss {l:.4f}, accuracy {a:.3f}, delay {dms.params['delay']}, "
                  f"{(time.time() - t_start) / (step + 1):.1f} s/step", flush=True)
        if np.isnan(l):
            print("loss is nan")
            break
        if l < TRAIN["loss_threshold"] and a > TRAIN["acc_threshold"] and dms.params["delay"][1] < 19000:
            # Curriculum: lengthen the delay (and, as in the paper, skip this update).
            dms.params["delay"][0] += TRAIN["delay_step"]
            dms.params["delay"][1] += TRAIN["delay_step"]
            print(f"step {step}: delay increased to {dms.params['delay']}", flush=True)
        else:
            optimizer.step()
            scheduler.step()
        if step % 50 == 0 or step == TRAIN["max_epochs"] - 1:
            w, gs, k = model.from_opt(opt)
            np.savez(os.path.join(args.out, "trained.npz"), input_weights=w.detach().cpu().numpy(), gS=gs.detach().cpu().numpy(),
                     k_minus=k.detach().cpu().numpy(), **{k_: np.asarray(v_) for k_, v_ in log.items()})
    print(f"training took {time.time() - t_start:.0f} s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=["evaluate", "train"], required=True)
    parser.add_argument("--params", default="paper")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--compile", type=int, default=1)
    parser.add_argument("--graph", type=int, default=1)
    parser.add_argument("--legacy", type=int, default=1, help="behave like the paper's Jaxley")
    parser.add_argument("--legacy-coupling", type=int, default=1,
                        help="couple branches without branchpoints, as Jaxley before 2024-08-01")
    parser.add_argument("--epochs", type=int, default=TRAIN["max_epochs"])
    args = parser.parse_args()
    TRAIN["max_epochs"] = args.epochs
    os.makedirs(args.out, exist_ok=True)
    torch.set_default_dtype(torch.float64)  # As in the paper (jax_enable_x64).
    device = torch.device(args.device)
    args.graph = bool(args.graph) and device.type == "cuda"
    use_legacy_axial_conductances(bool(args.legacy))
    use_legacy_branch_coupling(bool(args.legacy_coupling))
    args.variant = f"legacy{args.legacy}_coupling{args.legacy_coupling}"
    if args.mode == "evaluate":
        evaluate(args, device)
    else:
        train(args, device)


if __name__ == "__main__":
    main()
