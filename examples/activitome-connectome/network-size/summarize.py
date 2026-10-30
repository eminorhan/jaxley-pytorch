"""Tables and figures of the network-size experiment (see README.md).

Reads, for every network size N in `--sizes`, the runs of configs/n<N>_teacher.toml,
n<N>_students.toml and n<N>_control.toml in `--runs` (`n<N>_teacher/seed0`,
`n<N>_students/seed<k>`, `n<N>_control/seed0`), and with `--task-complexity` (that
experiment's runs directory) its token runs (`tokens_teacher`, `tokens_students`,
`tokens_control`, 300 neurons) as the first size; writes to `--out`:

  summary.md, summary.json  the tables;
  training_curves.png       the paper's Fig. 1c-e, one row per size;
  outcomes.png              every student's final task loss against its final activity
                            error, one panel per size;
  activity_by_size.png      every student's final activity error against the size.

Students that did not finish, or ended with a higher task loss than they started with
(their activity ran away), are left out of the statistics and figures and counted
(`jaxley_pytorch.training.report.excluded`).

Usage: python summarize.py --runs <runs dir> --sizes 600 1200 \
           --task-complexity <task-complexity runs dir> --out <dir>
"""
import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from jaxley_pytorch.training.report import (COLORS, INK, INK2, MUTED, SURFACE, excluded, load, neurons, spread,
                                            steps_in_thousands, style)

SOLVED = 1.02  # a student solves the task if its task loss is at most this many times the teacher's
LABELS = {"shuffled": "students, shuffled start", "control": "control: teacher's own start"}


def load_run(run_dir):
    path = os.path.join(run_dir, "result.json")
    return load(path) if os.path.exists(path) else None


def load_size(runs, prefix):
    """Teacher, students that finished without running away, number left out, and control
    of the runs `<prefix>_teacher`, `<prefix>_students` and `<prefix>_control` in `runs`
    (None for a missing run)."""
    dirs = sorted(glob.glob(os.path.join(runs, f"{prefix}_students", "seed*")))
    students = [s for s in map(load_run, dirs) if s is not None]
    kept = [s for s in students if not excluded(s)]
    teacher = load_run(os.path.join(runs, f"{prefix}_teacher", "seed0"))
    control = load_run(os.path.join(runs, f"{prefix}_control", "seed0"))
    return teacher, kept, len(dirs) - len(kept), control


def above_teacher(e):
    """Task loss above the teacher's, in %."""
    return 100 * (e["task_loss"] / e["teacher_task_loss"] - 1)


def student_row(n, students, n_excluded, control):
    base = students[0]["baselines"]
    first = [s["evals"][0] for s in students]
    last = [s["evals"][-1] for s in students]
    ratio = lambda evs, key: [e[key] / base[f"{key}_shuffled"] for e in evs]
    solved = sum(e["task_loss"] <= SOLVED * e["teacher_task_loss"] for e in last)
    control_ratio = f"{control['evals'][-1]['act_mse'] / base['act_mse_shuffled']:.3g}" if control else "-"
    entry = {"n": len(students), "excluded": n_excluded, "solved": solved, "initial": first, "final": last,
             "baselines": base, "act_ratio_initial": ratio(first, "act_mse"), "act_ratio_final": ratio(last, "act_mse")}
    row = (f"| {n:,} | {len(students)} | {n_excluded} | {solved} | {spread([above_teacher(e) for e in last], '{:.2f}')} | "
           f"{spread([e['output_vs_teacher'] for e in last])} | "
           f"{spread(ratio(first, 'act_mse'))} -> {spread(ratio(last, 'act_mse'))} | "
           f"{spread([e['act_corr_err'] for e in last])} ({base['act_corr_err_shuffled']:.3g}) | "
           f"{spread(ratio(last, 'act_mse_readout'))} | {spread(ratio(last, 'act_mse_other'))} | "
           f"{spread(ratio(last, 'gain_rmse'))} | {spread(ratio(last, 'bias_rmse'))} | {control_ratio} |")
    return entry, row


def training_curves(sizes, path):
    cols = [("task_loss", None, False, "task loss (bits/token)"),
            ("act_mse", "act_mse_shuffled", True, "error in activity (mean squared)"),
            ("gain_rmse", "gain_rmse_shuffled", False, "error in gains (RMS)"),
            ("bias_rmse", "bias_rmse_shuffled", False, "error in biases (RMS)")]
    fig, axes = plt.subplots(len(sizes), 4, figsize=(15, 3.3 * len(sizes) + 0.6), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for row, (n, (_, students, _, control)) in enumerate(sizes.items()):
        base = students[0]["baselines"]
        for col, (key, base_key, log_y, title) in enumerate(cols):
            ax = axes[row, col]
            style(ax)
            for s in students:
                ax.plot([e["step"] for e in s["evals"]], [e[key] for e in s["evals"]], color=COLORS["shuffled"],
                        lw=1.0, alpha=0.75, solid_capstyle="round")
            if control:
                ax.plot([e["step"] for e in control["evals"]], [e[key] for e in control["evals"]],
                        color=COLORS["control"], lw=2.0, solid_capstyle="round")
            ref = base[base_key] if base_key else students[0]["evals"][0]["teacher_task_loss"]
            ax.axhline(ref, color=INK if base_key is None else MUTED, lw=1.2)
            ax.annotate("teacher" if base_key is None else "shuffled neurons", (1.0, ref),
                        xycoords=("axes fraction", "data"), xytext=(-2, 3), textcoords="offset points",
                        ha="right", va="bottom", fontsize=8, color=INK2)
            if log_y:
                ax.set_yscale("log")
            steps_in_thousands(ax)
            ax.set_xlabel("training step", fontsize=8)
            ax.set_title(f"{n:,} neurons: {title}", fontsize=9, color=INK, loc="left")
    handles = [plt.Line2D([], [], color=COLORS[k], lw=2) for k in LABELS]
    fig.legend(handles, list(LABELS.values()), loc="upper center", ncol=2, frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / (3.3 * len(sizes) + 0.6)))
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def marker_legend(fig, fontsize):
    handles = [plt.Line2D([], [], ls="", marker=m, ms=7, color=COLORS[k], markeredgecolor=SURFACE)
               for k, m in [("shuffled", "o"), ("control", "D")]]
    fig.legend(handles, list(LABELS.values()), loc="upper center", ncol=2, frameon=False, fontsize=fontsize,
               labelcolor=INK)


def outcomes(sizes, path):
    fig, axes = plt.subplots(1, len(sizes), figsize=(max(5.2 * len(sizes), 9.0), 5.0), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for col, (n, (_, students, n_excluded, control)) in enumerate(sizes.items()):
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
        ax.set_title(f"{n:,} neurons", fontsize=10, color=INK, loc="left")
        if n_excluded:
            ax.text(0.99, 0.02, f"{n_excluded} student{'s' if n_excluded > 1 else ''} left out, not shown",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=INK2)
    marker_legend(fig, 9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def activity_by_size(sizes, path):
    """Every student's activity error after training (relative to the shuffled baseline)
    against the number of neurons."""
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    rng = np.random.default_rng(0)
    for n, (_, students, _, control) in sizes.items():
        base = students[0]["baselines"]
        r = [s["evals"][-1]["act_mse"] / base["act_mse_shuffled"] for s in students]
        ax.scatter(n * 2.0 ** rng.uniform(-0.025, 0.025, len(r)), r, s=36, color=COLORS["shuffled"],
                   edgecolor=SURFACE, linewidth=1.2, zorder=3)
        if control:
            ax.scatter([n], [control["evals"][-1]["act_mse"] / base["act_mse_shuffled"]], s=50, marker="D",
                       color=COLORS["control"], edgecolor=SURFACE, linewidth=1.2, zorder=3)
    for y, label, color in [(1.0, "shuffled neurons", MUTED), (0.1, "criterion: 0.1", INK2)]:
        ax.axhline(y, color=color, lw=1.0, zorder=1)
        ax.annotate(label, (1.0, y), xycoords=("axes fraction", "data"), xytext=(-2, 3), textcoords="offset points",
                    ha="right", fontsize=8, color=INK2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(sizes))
    ax.set_xticklabels([f"{n:,}" for n in sizes], fontsize=9, color=INK)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlim(min(sizes) / 2**0.4, max(sizes) * 2**0.4)
    ax.set_yscale("log")
    ax.set_xlabel("neurons", fontsize=8)
    ax.set_ylabel("activity error after training / shuffled-neuron baseline", fontsize=8)
    marker_legend(fig, 8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True)
    parser.add_argument("--sizes", nargs="+", type=int, required=True)
    parser.add_argument("--task-complexity", help="task-complexity's runs directory (its token runs: 300 neurons)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    arms = [(args.runs, f"n{n}") for n in args.sizes]
    if args.task_complexity:
        arms.append((args.task_complexity, "tokens"))
    sizes = {}
    for runs, prefix in arms:
        teacher, kept, n_excluded, control = load_size(runs, prefix)
        if teacher is None:
            print(f"{os.path.join(runs, prefix)}: no teacher result, left out", flush=True)
            continue
        sizes[neurons(teacher)] = (teacher, kept, n_excluded, control)
    sizes = dict(sorted(sizes.items()))

    summary, teacher_rows, student_rows = {}, [], []
    for n, (teacher, kept, n_excluded, control) in sizes.items():
        t = teacher["evals"][-1]
        teacher_rows.append(f"| {n:,} | {teacher['synapses']:,} | {t['task_loss']:.3f} bits/token, "
                            f"{t['val_bits_per_char']:.3f} bits/char | {t['participation_ratio']:.1f} | "
                            f"{t['components_90']} | {t['readout_variance_fraction']:.3f} | {t['mean_rate']:.3g} |")
        entry = {"synapses": teacher["synapses"], "teacher": t, "control_final": control["evals"][-1] if control else None}
        if kept:
            entry["students"], row = student_row(n, kept, n_excluded, control)
            student_rows.append(row)
        summary[n] = entry

    lines = ["## Teachers", "",
             "| neurons | synapses | task loss (validation) | participation ratio | components for 90% | "
             "variance in the readout's directions | mean rate |",
             "|---|---|---|---|---|---|---|", *teacher_rows, "",
             "## Students", "",
             "Median [min, max] over the students of a size, all started from the teacher's gains and biases permuted "
             "(gains halved). Left out: did not finish, or ended with a higher task loss than at the start. Errors are "
             "relative to the shuffled baseline unless noted. Solved: task loss within 2% of the teacher's. Control: the "
             "student started from the teacher's own gains and biases.", "",
             "| neurons | students | left out | solved | task loss above the teacher's (%) | KL from the teacher "
             "(bits/token) | activity error, before -> after | 1 - mean abs r (shuffled) | activity error in the "
             "readout's directions | ... in the other directions | gain error | bias error | control: activity error "
             "after |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|", *student_rows, ""]
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(text)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    plotted = {n: v for n, v in sizes.items() if v[1]}
    if plotted:
        training_curves(plotted, os.path.join(args.out, "training_curves.png"))
        outcomes(plotted, os.path.join(args.out, "outcomes.png"))
        activity_by_size(plotted, os.path.join(args.out, "activity_by_size.png"))


if __name__ == "__main__":
    main()
