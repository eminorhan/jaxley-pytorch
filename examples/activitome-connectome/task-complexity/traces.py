"""Example traces (the paper's Fig. 1f,g and 5c), simulated on the CPU; writes
`traces_<task>.png` to `--out`.

  cycling  the readout, and the activity of one excitatory and one inhibitory neuron, in
           the teacher and in the best student of each start (shuffled, random: lowest
           final task loss), on one trial of each input;
  chars, tokens
           over 64 tokens of validation text (after 32 tokens from rest): the probability
           of the actual next token, and the activity (rates averaged over each token's
           window) of one excitatory and one inhibitory neuron, in the teacher and in the
           best shuffled-start student before and after training.

The two neurons are the excitatory and the inhibitory one whose activity varies most in
the teacher. The task settings are those of the task's teacher config in configs/.

Usage: python traces.py --runs <runs dir> --out <dir> [--task cycling]
"""
import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from jaxley_pytorch.training import init, rate_network, tasks
from jaxley_pytorch.training.config import load_config
from jaxley_pytorch.training.report import COLORS, INK, INK2, MUTED, SURFACE, excluded, load, style

STARTS = {"shuffled": "{task}_students/seed*", "random": "{task}_students_random/seed*"}
SHOWN = 64  # text tokens shown
CONFIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")


def task_of(name):
    """The task of `configs/<name>_teacher.toml`."""
    task = load_config(os.path.join(CONFIGS, f"{name}_teacher.toml")).task
    return tasks.TASKS[task.name](**task.kwargs)


def best_student(runs, task, pattern):
    """Directory and final task loss of the kept student with the lowest final task loss."""
    best = None
    for d in sorted(glob.glob(os.path.join(runs, pattern.format(task=task)))):
        path = os.path.join(d, "result.json")
        if os.path.exists(path):
            s = load(path)
            if not excluded(s) and (best is None or s["evals"][-1]["task_loss"] < best[1]):
                best = (d, s["evals"][-1]["task_loss"])
    return best


def trained(teacher, run_dir):
    st = torch.load(os.path.join(run_dir, "student.pt"), map_location="cpu", weights_only=True)
    io = {k: v.detach() for k, v in teacher.io.items()}
    return rate_network.RateNetwork(teacher.sim, teacher.W.detach(), st["g"], st["b"], io), int(st["seed"])


def two_neurons(a_t, n_exc):
    """(name, index) of the excitatory and the inhibitory neuron whose activity varies most;
    `a_t` is (..., n)."""
    spread = a_t.reshape(-1, a_t.shape[-1]).var(0)
    return [("excitatory", int(np.argmax(spread[:n_exc]))), ("inhibitory", n_exc + int(np.argmax(spread[n_exc:])))]


def cycling_traces(args, teacher, conn):
    students = {}
    for start, pattern in STARTS.items():
        best = best_student(args.runs, "cycling", pattern)
        if best is not None:
            students[start] = (os.path.basename(best[0]), best[1], trained(teacher, best[0])[0])
    task = task_of("cycling")
    dt_tau = teacher.sim.dt / rate_network.TAU
    inputs, targets = tasks.cycling_trials(task.trials_seed, task.trials, task.trial_steps, dt_tau, task.omega)
    trials = [int(np.flatnonzero(inputs[:, :, c].sum(1) > 0)[0]) for c in (0, 1)]  # one trial per input
    inp = torch.as_tensor(inputs[trials])

    def outputs(model):
        with torch.no_grad():
            z, a, _ = task.run(model, inp, model.sim.rest_states(len(trials)))
        return {"z_s": z, "a_s": a.transpose(1, 2)}

    runs = {"teacher": outputs(teacher)}
    for start, (_, _, model) in students.items():
        runs[start] = outputs(model)
    neurons = two_neurons(runs["teacher"]["a_s"].numpy(), conn["n_exc"])

    t = np.arange(task.trial_steps) * dt_tau  # in units of tau
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.4), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    lines = [("teacher", INK, 1.6)] + [(start, COLORS[start], 1.4) for start in students]
    for row, trial in enumerate(trials):
        pulse = t[int(np.flatnonzero(inputs[trial].sum(1) > 0)[0])]
        for col, (name, neuron) in enumerate([("readout", None)] + neurons):
            ax = axes[row, col]
            style(ax)
            if neuron is None:
                ax.plot(t, targets[trial], color=MUTED, lw=3.0, alpha=0.5, solid_capstyle="round")
            for key, color, lw in lines:
                y = runs[key]["z_s"][row].numpy() if neuron is None else runs[key]["a_s"][row, :, neuron].numpy()
                ax.plot(t, y, color=color, lw=lw, solid_capstyle="round")
            ax.axvline(pulse, color=MUTED, lw=0.8)
            title = name if neuron is None else f"{name} neuron {neuron}"
            ax.set_title(f"input {row + 1}: {title}", fontsize=9, color=INK, loc="left")
            ax.set_ylabel("readout" if neuron is None else "rate", fontsize=8)
            if row == 1:
                ax.set_xlabel("time (tau)", fontsize=8)
    handles = [plt.Line2D([], [], color=MUTED, lw=3.0, alpha=0.5)] + [plt.Line2D([], [], color=c, lw=2)
                                                                      for _, c, _ in lines]
    labels = ["target", "teacher"] + [f"best {start}-start student ({name}, task loss {loss:.2g})"
                                      for start, (name, loss, _) in students.items()]
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=9, labelcolor=INK)
    fig.text(0.01, 0.005, "Vertical line: the input pulse. Neurons: the excitatory and the inhibitory one whose "
             "activity varies most in the teacher.", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    return fig, {"trials": trials, "neurons": neurons, "students": {k: v[:2] for k, v in students.items()}}


def token_labels(task, data, ids):
    """Printable label of every token id."""
    if task == "chars":
        chars = tasks.load_meta(data)["chars"]
        text = ["<eos>" if i == 0 else chars[i - 1] for i in ids]
    else:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(os.path.join(data, "tokenizer.json"))
        text = ["<eos>" if i == 0 else tok.decode([int(i)]) for i in ids]
    return [s.replace("\n", "\\n").replace(" ", "·") for s in text]


def text_traces(args, task, teacher, conn):
    best = best_student(args.runs, task, STARTS["shuffled"])
    after, seed = trained(teacher, best[0])
    before = init.make_student(teacher, conn, "shuffled", seed)
    text = task_of(task)
    data = text.data
    meta = tasks.load_meta(data)
    seg = tasks.eval_segments(os.path.join(data, "val.bin"), meta["dtype"], args.eval_segments, args.eval_burn + SHOWN)[0]
    tok = torch.as_tensor(seg[None, :-1])
    nxt = seg[1 + args.eval_burn:]
    runs = {}
    with torch.no_grad():
        for key, model in (("teacher", teacher), ("after", after), ("before", before)):
            logits, feats, _ = text.run(model, tok, model.sim.rest_states(1))
            prob = F.softmax(logits[0, args.eval_burn:].float(), dim=-1)[torch.arange(SHOWN), torch.as_tensor(nxt)]
            runs[key] = (prob.numpy(), feats[0, args.eval_burn:].numpy())
    neurons = two_neurons(runs["teacher"][1], conn["n_exc"])

    x = np.arange(SHOWN)
    fig, axes = plt.subplots(3, 1, figsize=(15, 8.2), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    lines = [("before", MUTED, 1.4), ("teacher", INK, 1.8), ("after", COLORS["shuffled"], 1.4)]
    panels = [("probability of the actual next token", None)] + [(f"{name} neuron {i}: rate", i) for name, i in neurons]
    for ax, (title, neuron) in zip(axes, panels):
        style(ax)
        for key, color, lw in lines:
            y = runs[key][0] if neuron is None else runs[key][1][:, neuron]
            ax.plot(x, y, color=color, lw=lw, solid_capstyle="round", marker="o", ms=2.5)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(token_labels(task, data, seg[1 + args.eval_burn - 1:-1]), fontsize=7,
                             rotation=90 if task == "tokens" else 0, color=INK2)
    axes[-1].set_xlabel("input token", fontsize=8)
    handles = [plt.Line2D([], [], color=c, lw=2) for _, c, _ in lines]
    labels = [f"student before training ({os.path.basename(best[0])}: shuffled start)", "teacher",
              f"same student after training (task loss {best[1]:.4g} bits/token)"]
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9, labelcolor=INK)
    fig.text(0.01, 0.005, f"{task}: {SHOWN} tokens of validation text after {args.eval_burn} from rest; activity = rate "
             "averaged over each token's window. Neurons: the excitatory and the inhibitory one whose activity "
             "varies most in the teacher.", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    return fig, {"student": os.path.basename(best[0]), "seed": seed, "neurons": neurons}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--task", choices=["cycling", "chars", "tokens"], default="cycling")
    parser.add_argument("--eval-segments", type=int, default=64)
    parser.add_argument("--eval-burn", type=int, default=32)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_default_dtype(torch.float32)
    teacher, conn, _ = rate_network.load_network(os.path.join(args.runs, f"{args.task}_teacher", "seed0"),
                                                 torch.device("cpu"), torch.float32, False)
    if args.task == "cycling":
        fig, info = cycling_traces(args, teacher, conn)
    else:
        fig, info = text_traces(args, args.task, teacher, conn)
    fig.savefig(os.path.join(args.out, f"traces_{args.task}.png"), dpi=150, facecolor=SURFACE)
    print(info)


if __name__ == "__main__":
    main()
