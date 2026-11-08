"""Tables and figures of the complex-neurons experiment's stage 2 (see README.md), and of
other experiments with its runs (../cell-types).

Reads, for every model in `--models`, the runs of configs/<model>_n1024_teacher.toml,
<model>_n1024_students.toml and <model>_n1024_control.toml in `--runs`
(`<model>_n1024_teacher/seed0`, `<model>_n1024_students/seed<k>`,
`<model>_n1024_control/seed0`), or with `--group LABEL RUNS PREFIX` (repeated) the runs
<PREFIX>_teacher, <PREFIX>_students and <PREFIX>_control in RUNS of every group, in
that order; writes to `--out`:

  summary.md, summary.json  the tables;
  training_curves.png       task loss, activity error and spike-count correlation during
                            training, one row per model (group);
  parameters.png            the error in every single-neuron parameter during training,
                            one row per model (group);
  outcomes.png              every student's final task loss against its final activity
                            error, one panel per model (group);
  by_group.png              every student's final error in the activity and in every
                            single-neuron parameter, per model (group).

Errors are relative to the shuffled-neuron baseline (the teacher against a permutation of
its own neurons) unless noted. Students that did not finish, or ended with a higher task
loss than they started with, are left out of the statistics and figures and counted
(`jaxley_pytorch.training.report.excluded`).

Usage: python summarize.py --runs <runs dir> --out <dir> [--models pospischil hh]
       python summarize.py --group LABEL RUNS PREFIX [--group ...] --out <dir>
"""
import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from jaxley_pytorch.training.report import (COLORS, INK, INK2, MUTED, SURFACE, excluded, load, spread,
                                            steps_in_thousands, style, teacher_run)

NAMES = {"pospischil": "Pospischil", "hh": "Hodgkin-Huxley"}
PARAMS = ["gNa", "gK", "gKm", "gLeak", "bias"]  # the single-neuron parameters (HH has no gKm)
LABELS = {"shuffled": "students, shuffled start", "control": "control: teacher's own start"}


def load_group(runs, prefix):
    """Teacher, students that finished without running away, number left out, control."""
    teacher = load(os.path.join(teacher_run(runs, prefix), "result.json"))
    students = [load(p) for p in sorted(glob.glob(os.path.join(runs, f"{prefix}_students", "seed*", "result.json")))]
    kept = [s for s in students if not excluded(s)]
    path = os.path.join(runs, f"{prefix}_control", "seed0", "result.json")
    control = load(path) if os.path.exists(path) else None
    return teacher, kept, len(students) - len(kept), control


def above_teacher(e):
    """Task loss above the teacher's, in %."""
    return 100 * (e["task_loss"] / e["teacher_task_loss"] - 1)


def tables(models):
    rows_t = ["| model | task loss (validation) | firing rate | participation ratio | components for 90% |",
              "|---|---|---|---|---|"]
    head = ("| model | students | left out | task loss above the teacher's (%) | KL from the teacher (bits/token) | "
            "activity error, before -> after | 1 - mean abs r (shuffled) | spike-count correlation error (shuffled) | "
            + " | ".join(f"{p} error" for p in PARAMS) + " | control: activity error after |")
    rows_s = [head, "|---" * (9 + len(PARAMS)) + "|"]
    out = {}
    for label, (teacher, students, n_excluded, control) in models.items():
        t = teacher["final"]
        rows_t.append(f"| {label} | {t['task_loss']:.3f} bits/token | {t['firing_rate_hz']:.1f} Hz | "
                      f"{t['participation_ratio']:.1f} | {t['components_90']} |")
        base = students[0]["baselines"]
        first = [s["evals"][0] for s in students]
        last = [s["evals"][-1] for s in students]
        ratio = lambda evs, key: [e[key] / base[f"{key}_shuffled"] for e in evs]
        params = [spread(ratio(last, f"{p}_rmse")) if f"{p}_rmse" in last[0] else "-" for p in PARAMS]
        control_act = (f"{control['evals'][-1]['act_mse'] / base['act_mse_shuffled']:.3g}" if control else "-")
        rows_s.append(
            f"| {label} | {len(students)} | {n_excluded} | {spread([above_teacher(e) for e in last])} | "
            f"{spread([e['output_vs_teacher'] for e in last])} | "
            f"{spread(ratio(first, 'act_mse'))} -> {spread(ratio(last, 'act_mse'))} | "
            f"{spread([e['act_corr_err'] for e in last])} ({base['act_corr_err_shuffled']:.3g}) | "
            f"{spread([e['spike_corr_err'] for e in last])} ({base['spike_corr_err_shuffled']:.3g}) | "
            + " | ".join(params) + f" | {control_act} |")
        out[label] = {"teacher": t, "students": len(students), "excluded": n_excluded, "baselines": base,
                      "initial": first, "final": last, "control_final": control["evals"][-1] if control else None}
    text = "\n".join(
        ["# complex-neurons, stage 2 (1,024 neurons)", "", "## Teachers", ""] + rows_t
        + ["", "## Students", "",
           "Median [min, max] over the students of a model, all started from the teacher's single-neuron parameters "
           "permuted within the excitatory and within the inhibitory neurons. Left out: did not finish, or ended with "
           "a higher task loss than at the start. Errors are relative to the shuffled-neuron baseline unless noted "
           "(the readout reads every direction of the activity, so there is no split into the readout's and the other "
           "directions). Control: the student started from the teacher's own parameters.", ""]
        + rows_s)
    return text, out


def curves(ax, students, control, key, base_key, base, log_y=False):
    for s in students:
        ax.plot([e["step"] for e in s["evals"]], [e[key] for e in s["evals"]], color=COLORS["shuffled"], lw=1.0,
                alpha=0.75, solid_capstyle="round")
    if control:
        ax.plot([e["step"] for e in control["evals"]], [e[key] for e in control["evals"]], color=COLORS["control"],
                lw=2.0, solid_capstyle="round")
    ref = base[base_key] if base_key else students[0]["evals"][0]["teacher_task_loss"]
    ax.axhline(ref, color=INK if base_key is None else MUTED, lw=1.2)
    # The label goes on the side of the line away from where the students end.
    below = sorted(s["evals"][-1][key] for s in students)[len(students) // 2] > ref
    ax.annotate("teacher" if base_key is None else "shuffled neurons", (1.0, ref), xycoords=("axes fraction", "data"),
                xytext=(-2, -3 if below else 3), textcoords="offset points", ha="right",
                va="top" if below else "bottom", fontsize=8, color=INK2)
    if log_y:
        ax.set_yscale("log")
    else:  # room for the label above a line at the top of the panel
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, max(hi, ref + 0.1 * (hi - lo)))
    steps_in_thousands(ax)
    ax.set_xlabel("training step", fontsize=8)


def line_legend(fig):
    handles = [plt.Line2D([], [], color=COLORS[k], lw=2) for k in LABELS]
    fig.legend(handles, list(LABELS.values()), loc="upper center", ncol=len(LABELS), frameon=False, fontsize=9,
               labelcolor=INK)


def training_curves(models, path):
    cols = [("task_loss", None, False, "task loss (bits/token)"),
            ("act_mse", "act_mse_shuffled", True, "error in activity (mean squared)"),
            ("spike_corr_err", "spike_corr_err_shuffled", False, "spike counts: 1 - correlation")]
    fig, axes = plt.subplots(len(models), len(cols), figsize=(12, 3.3 * len(models) + 0.6), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for row, (label, (_, students, _, control)) in enumerate(models.items()):
        base = students[0]["baselines"]
        for col, (key, base_key, log_y, title) in enumerate(cols):
            ax = axes[row, col]
            style(ax)
            curves(ax, students, control, key, base_key, base, log_y)
            ax.set_title(f"{label}: {title}", fontsize=9, color=INK, loc="left")
    line_legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / (3.3 * len(models) + 0.6)))
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def parameters(models, path):
    fig, axes = plt.subplots(len(models), len(PARAMS), figsize=(16, 3.3 * len(models) + 0.6), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for row, (label, (_, students, _, control)) in enumerate(models.items()):
        base = students[0]["baselines"]
        for col, p in enumerate(PARAMS):
            ax = axes[row, col]
            style(ax)
            if f"{p}_rmse" not in students[0]["evals"][-1]:
                ax.text(0.5, 0.5, f"{label}: these neurons have no {p}", transform=ax.transAxes, ha="center",
                        va="center", fontsize=8, color=INK2)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            curves(ax, students, control, f"{p}_rmse", f"{p}_rmse_shuffled", base)
            ax.set_title(f"{label}: error in {p} (RMS)", fontsize=9, color=INK, loc="left")
    line_legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / (3.3 * len(models) + 0.6)))
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def outcomes(models, path):
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 5.0), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for col, (label, (_, students, n_excluded, control)) in enumerate(models.items()):
        ax = axes[0, col]
        style(ax)
        base = students[0]["baselines"]
        last = [s["evals"][-1] for s in students]
        ax.scatter([above_teacher(e) for e in last], [e["act_mse"] / base["act_mse_shuffled"] for e in last], s=42,
                   color=COLORS["shuffled"], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        if control:
            e = control["evals"][-1]
            ax.scatter([above_teacher(e)], [e["act_mse"] / base["act_mse_shuffled"]], s=60, marker="D",
                       color=COLORS["control"], edgecolor=SURFACE, linewidth=1.5, zorder=4)
        ax.axhline(1.0, color=MUTED, lw=1.2, zorder=1)
        ax.annotate("shuffled neurons", (1.0, 1.0), xycoords=("axes fraction", "data"), xytext=(-2, 3),
                    textcoords="offset points", ha="right", fontsize=8, color=INK2)
        ax.axvline(0.0, color=INK, lw=1.2, zorder=1)
        ax.annotate("teacher", (0.0, 1.0), xycoords=("data", "axes fraction"), xytext=(3, -12),
                    textcoords="offset points", fontsize=8, color=INK2)
        ax.set_yscale("log")
        ax.set_xlabel("final task loss above the teacher's (%)", fontsize=8)
        ax.set_ylabel("final activity error / shuffled-neuron baseline", fontsize=8)
        ax.set_title(label, fontsize=10, color=INK, loc="left")
        if n_excluded:
            ax.text(0.99, 0.02, f"{n_excluded} student{'s' if n_excluded > 1 else ''} left out, not shown",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=INK2)
    handles = [plt.Line2D([], [], ls="", marker=m, ms=7, color=COLORS[k], markeredgecolor=SURFACE)
               for k, m in (("shuffled", "o"), ("control", "D"))]
    fig.legend(handles, list(LABELS.values()), loc="upper center", ncol=2, frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def by_group(models, path):
    """Every student's final error in the activity (log scale) and in every single-neuron
    parameter, relative to the shuffled-neuron baseline, per model (group)."""
    panels = [("act_mse", "activity (mean squared)", True)] + [(f"{p}_rmse", p, False) for p in PARAMS]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.3 * len(panels) + 1.4 * len(models), 4.6), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    rng = np.random.default_rng(0)
    x = np.arange(len(models))
    for ax, (key, title, log_y) in zip(axes[0], panels):
        style(ax)
        for i, (_, students, _, control) in enumerate(models.values()):
            base = students[0]["baselines"]
            if key not in students[0]["evals"][-1]:
                continue
            r = [s["evals"][-1][key] / base[f"{key}_shuffled"] for s in students]
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(r)), r, s=30, color=COLORS["shuffled"], edgecolor=SURFACE,
                       linewidth=1.0, zorder=3)
            if control:
                ax.scatter([i], [control["evals"][-1][key] / base[f"{key}_shuffled"]], s=46, marker="D",
                           color=COLORS["control"], edgecolor=SURFACE, linewidth=1.0, zorder=4)
        ax.axhline(1.0, color=MUTED, lw=1.2, zorder=1)
        ax.annotate("shuffled neurons", (1.0, 1.0), xycoords=("axes fraction", "data"), xytext=(-2, 3),
                    textcoords="offset points", ha="right", fontsize=7.5, color=INK2)
        if log_y:
            ax.set_yscale("log")
        else:
            ax.set_ylim(bottom=0.0)
        ax.set_xticks(x)
        ax.set_xticklabels(list(models), fontsize=8, rotation=30, ha="right", color=INK)
        ax.set_xlim(-0.5, len(models) - 0.5)
        ax.set_title(f"error in {title}", fontsize=9, color=INK, loc="left")
    axes[0, 0].set_ylabel("final error / shuffled-neuron baseline", fontsize=8)
    handles = [plt.Line2D([], [], ls="", marker=m, ms=7, color=COLORS[k], markeredgecolor=SURFACE)
               for k, m in (("shuffled", "o"), ("control", "D"))]
    fig.legend(handles, list(LABELS.values()), loc="upper center", ncol=2, frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", help="the runs directory of --models")
    parser.add_argument("--out", required=True)
    parser.add_argument("--models", nargs="+", choices=list(NAMES), default=list(NAMES))
    parser.add_argument("--group", nargs=3, action="append", metavar=("LABEL", "RUNS", "PREFIX"),
                        help="instead of --models: a group of runs (repeat for more)")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.group:
        groups = args.group
    elif args.runs:
        groups = [(NAMES[m], args.runs, f"{m}_n1024") for m in args.models]
    else:
        parser.error("give --runs (with --models) or --group")
    models = {label: load_group(runs, prefix) for label, runs, prefix in groups}
    text, out = tables(models)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(text, flush=True)
    training_curves(models, os.path.join(args.out, "training_curves.png"))
    parameters(models, os.path.join(args.out, "parameters.png"))
    outcomes(models, os.path.join(args.out, "outcomes.png"))
    by_group(models, os.path.join(args.out, "by_group.png"))


if __name__ == "__main__":
    main()
