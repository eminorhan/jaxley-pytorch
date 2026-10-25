"""Tables and figures of the task-complexity experiment (see README.md).

Reads, per task (`--tasks`, default cycling chars tokens; missing ones are skipped), the
result.json of `<runs>/<task>/teacher`, of the students `student_seed*` (shuffled
start), `student_random_seed*` (random start) and of `student_control`, and writes to
`--out`:

  summary.md, summary.json  the tables;
  training_curves.png       the paper's Fig. 1c-e for every task (one row per task);
  outcomes.png              every student's final task loss against its final activity
                            error (one panel per task);
  activity_by_task.png      every student's final activity error, one column per task.

Students that did not finish, or ended with a higher task loss than they started with
(their activity ran away), are left out of the statistics and figures and counted.

Usage: python summarize.py --runs <runs dir> --out <dir> [--tasks cycling]
"""
import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TASKS = ["cycling", "chars", "tokens"]
LOSS = {"cycling": "task loss (MSE)", "chars": "task loss (bits/token)", "tokens": "task loss (bits/token)"}
# A student solves the task if its task loss is at most this many times the teacher's.
SOLVED = {"cycling": 2.0, "chars": 1.02, "tokens": 1.02}
STARTS = {"shuffled": "student_seed*", "random": "student_random_seed*"}

# Colors: the dataviz reference palette (categorical slots 1-3 and chart ink).
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLORS = {"shuffled": "#2a78d6", "random": "#eb6834", "control": "#1baf7a"}
LABELS = {"shuffled": "students, shuffled start", "random": "students, random start",
          "control": "control: teacher's own start"}


def load(path):
    with open(path) as f:
        return json.load(f)


def excluded(s):
    """Did not finish, or ended with a higher task loss than it started with."""
    first, last = s["evals"][0]["task_loss"], s["evals"][-1]["task_loss"]
    return "final" not in s or not np.isfinite(last) or last > first


def spread(values, fmt="{:.3g}"):
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return "-"
    return (fmt + " [" + fmt + ", " + fmt + "]").format(np.median(v), v.min(), v.max())


def steps_in_thousands(ax):
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5, integer=True))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1000:g}k" if x else "0"))


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def student_row(task, start, students, n_excluded, control, base):
    first = [s["evals"][0] for s in students]
    last = [s["evals"][-1] for s in students]
    ratio = lambda evs, key: [e[key] / base[f"{key}_shuffled"] for e in evs]
    solved = sum(e["task_loss"] <= SOLVED[task] * e["teacher_task_loss"] for e in last)
    control_ratio = f"{control['evals'][-1]['act_mse'] / base['act_mse_shuffled']:.3g}" if control else "-"
    entry = {"n": len(students), "excluded": n_excluded, "solved": solved, "initial": first, "final": last,
             "act_ratio_initial": ratio(first, "act_mse"), "act_ratio_final": ratio(last, "act_mse")}
    row = (f"| {task} | {start} | {len(students)} | {n_excluded} | {solved} | "
           f"{spread([e['task_loss'] for e in last])} ({last[0]['teacher_task_loss']:.3g}) | "
           f"{spread([e['output_vs_teacher'] for e in last])} | "
           f"{spread(ratio(first, 'act_mse'), '{:.3g}')} -> {spread(ratio(last, 'act_mse'), '{:.3g}')} | "
           f"{spread([e['act_corr_err'] for e in last], '{:.3g}')} ({base['act_corr_err_shuffled']:.3g}) | "
           f"{spread(ratio(last, 'act_mse_readout'), '{:.3g}')} | {spread(ratio(last, 'act_mse_other'), '{:.3g}')} | "
           f"{spread(ratio(last, 'gain_rmse'), '{:.3g}')} | {spread(ratio(last, 'bias_rmse'), '{:.3g}')} | "
           f"{control_ratio} |")
    return entry, row


def load_students(runs, task, pattern):
    """(students that finished without running away, number left out)."""
    dirs = sorted(glob.glob(os.path.join(runs, task, pattern)))
    results = [load(os.path.join(d, "result.json")) for d in dirs if os.path.exists(os.path.join(d, "result.json"))]
    kept = [s for s in results if not excluded(s)]
    return kept, len(dirs) - len(kept)


def training_curves(curves, path):
    cols = [("task_loss", None, True), ("act_mse", "act_mse_shuffled", True),
            ("gain_rmse", "gain_rmse_shuffled", False), ("bias_rmse", "bias_rmse_shuffled", False)]
    titles = [None, "error in activity (mean squared)", "error in gains (RMS)", "error in biases (RMS)"]
    fig, axes = plt.subplots(len(curves), 4, figsize=(15, 3.3 * len(curves) + 0.6), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for row, (task, (groups, control, base, teacher_loss)) in enumerate(curves.items()):
        for col, (key, base_key, log_y) in enumerate(cols):
            ax = axes[row, col]
            style(ax)
            for start, (students, _) in groups.items():
                for s in students:
                    ax.plot([e["step"] for e in s["evals"]], [e[key] for e in s["evals"]], color=COLORS[start],
                            lw=1.0, alpha=0.75, solid_capstyle="round")
            if control:
                ax.plot([e["step"] for e in control["evals"]], [e[key] for e in control["evals"]],
                        color=COLORS["control"], lw=2.0, solid_capstyle="round")
            ref = base[base_key] if base_key else teacher_loss
            ax.axhline(ref, color=INK if base_key is None else MUTED, lw=1.2)
            ax.annotate("teacher" if base_key is None else "shuffled neurons", (1.0, ref),
                        xycoords=("axes fraction", "data"), xytext=(-2, 3), textcoords="offset points",
                        ha="right", va="bottom", fontsize=8, color=INK2)
            if log_y and (task == "cycling" or key != "task_loss"):
                ax.set_yscale("log")
            steps_in_thousands(ax)
            ax.set_xlabel("training step", fontsize=8)
            ax.set_title(f"{task}: {titles[col] or LOSS[task]}", fontsize=9, color=INK, loc="left")
    handles = [plt.Line2D([], [], color=COLORS[k], lw=2) for k in ["shuffled", "random", "control"]]
    fig.legend(handles, [LABELS[k] for k in ["shuffled", "random", "control"]], loc="upper center", ncol=3,
               frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / (3.3 * len(curves) + 0.6)))
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def outcomes(curves, path):
    fig, axes = plt.subplots(1, len(curves), figsize=(max(5.2 * len(curves), 9.0), 5.0), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for col, (task, (groups, control, base, teacher_loss)) in enumerate(curves.items()):
        ax = axes[0, col]
        style(ax)
        notes = []
        # Cycling: the task loss itself (log scale); text: how far above the teacher's, in %.
        x_of = (lambda v: v) if task == "cycling" else (lambda v: 100 * (v / teacher_loss - 1))
        for start, (students, n_excluded) in groups.items():
            last = [s["evals"][-1] for s in students]
            ax.scatter([x_of(e["task_loss"]) for e in last], [e["act_mse"] / base["act_mse_shuffled"] for e in last],
                       s=42, color=COLORS[start], edgecolor=SURFACE, linewidth=1.5, zorder=3)
            if n_excluded:
                notes.append(f"{n_excluded} {start}-start student{'s' if n_excluded > 1 else ''} ran away, not shown")
        if control:
            e = control["evals"][-1]
            ax.scatter([x_of(e["task_loss"])], [e["act_mse"] / base["act_mse_shuffled"]], s=60, marker="D",
                       color=COLORS["control"], edgecolor=SURFACE, linewidth=1.5, zorder=4)
        ax.axhline(1.0, color=MUTED, lw=1.2, zorder=1)
        ax.annotate("shuffled neurons", (1.0, 1.0), xycoords=("axes fraction", "data"), xytext=(-2, 3),
                    textcoords="offset points", ha="right", fontsize=8, color=INK2)
        ax.axvline(x_of(teacher_loss), color=INK, lw=1.2, zorder=1)
        ax.annotate("teacher", (x_of(teacher_loss), 1.0), xycoords=("data", "axes fraction"), xytext=(3, -12),
                    textcoords="offset points", fontsize=8, color=INK2)
        if task == "cycling":
            ax.set_xscale("log")
            ax.set_xlabel(f"final {LOSS[task]}", fontsize=8)
        else:
            ax.set_xlabel("final task loss above the teacher's (%)", fontsize=8)
        ax.set_yscale("log")
        ax.set_ylabel("final activity error / shuffled-neuron baseline", fontsize=8)
        ax.set_title(task, fontsize=10, color=INK, loc="left")
        if notes:
            ax.text(0.99, 0.02, "\n".join(notes), transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
                    color=INK2)
    handles = [plt.Line2D([], [], ls="", marker=m, ms=7, color=COLORS[k], markeredgecolor=SURFACE)
               for k, m in [("shuffled", "o"), ("random", "o"), ("control", "D")]]
    fig.legend(handles, [LABELS[k] for k in ["shuffled", "random", "control"]], loc="upper center", ncol=3,
               frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def activity_by_task(curves, path):
    """Every student's activity error after training (relative to the shuffled baseline),
    one column per task."""
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    rng = np.random.default_rng(0)
    for i, (task, (groups, control, base, _)) in enumerate(curves.items()):
        both = len(groups) > 1
        for start, (students, _) in groups.items():
            r = [s["evals"][-1]["act_mse"] / base["act_mse_shuffled"] for s in students]
            x = i + (0.0 if not both else (-0.07 if start == "shuffled" else 0.07)) + rng.uniform(-0.025, 0.025, len(r))
            ax.scatter(x, r, s=36, color=COLORS[start], edgecolor=SURFACE, linewidth=1.2, zorder=3)
        if control:
            e = control["evals"][-1]
            ax.scatter([i], [e["act_mse"] / base["act_mse_shuffled"]], s=50, marker="D",
                       color=COLORS["control"], edgecolor=SURFACE, linewidth=1.2, zorder=3)
    for y, label, color in [(1.0, "shuffled neurons", MUTED), (0.1, "criterion: 0.1", INK2)]:
        ax.axhline(y, color=color, lw=1.0, zorder=1)
        ax.annotate(label, (1.0, y), xycoords=("axes fraction", "data"), xytext=(-2, 3), textcoords="offset points",
                    ha="right", fontsize=8, color=INK2)
    ax.set_yscale("log")
    ax.set_xticks(range(len(curves)))
    ax.set_xticklabels(list(curves), fontsize=9, color=INK)
    ax.set_xlim(-0.6, len(curves) - 0.4)
    ax.set_ylabel("activity error after training / shuffled-neuron baseline", fontsize=8)
    handles = [plt.Line2D([], [], ls="", marker=m, ms=7, color=COLORS[k], markeredgecolor=SURFACE)
               for k, m in [("shuffled", "o"), ("random", "o"), ("control", "D")]]
    fig.legend(handles, [LABELS[k] for k in ["shuffled", "random", "control"]], loc="upper center", ncol=3,
               frameon=False, fontsize=8.5, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    summary, teacher_rows, student_rows, curves = {}, [], [], {}
    for task in args.tasks:
        tpath = os.path.join(args.runs, task, "teacher", "result.json")
        groups = {start: load_students(args.runs, task, pattern) for start, pattern in STARTS.items()}
        cpath = os.path.join(args.runs, task, "student_control", "result.json")
        control = load(cpath) if os.path.exists(cpath) else None
        if not os.path.exists(tpath) and not any(students for students, _ in groups.values()):
            continue
        entry = {"control_final": control["evals"][-1] if control else None}
        if os.path.exists(tpath):
            t = load(tpath)["evals"][-1]
            entry["teacher"] = t
            loss = (f"MSE {t['task_loss']:.3g}" if task == "cycling" else
                    f"{t['task_loss']:.3f} bits/token, {t['val_bits_per_char']:.3f} bits/char")
            teacher_rows.append(f"| {task} | {loss} | {t['participation_ratio']:.1f} | {t['components_90']} | "
                                f"{t['readout_rank']} | {t['readout_variance_fraction']:.3f} | {t['mean_rate']:.3g} |")
        kept = {start: g for start, g in groups.items() if g[0]}
        if not kept:
            summary[task] = entry
            continue
        some = next(iter(kept.values()))[0][0]
        base, teacher_loss = some["baselines"], some["evals"][0]["teacher_task_loss"]
        entry["baselines"] = base
        for start, (students, n_excluded) in kept.items():
            entry[start], row = student_row(task, start, students, n_excluded, control, base)
            student_rows.append(row)
        curves[task] = (kept, control, base, teacher_loss)
        summary[task] = entry

    lines = ["## Teachers", "",
             "| task | task loss (validation) | participation ratio | components for 90% | readout rank | "
             "variance in the readout's directions | mean rate |",
             "|---|---|---|---|---|---|---|", *teacher_rows, "",
             "## Students", "",
             "Median [min, max] over the students. Start: shuffled = the teacher's gains and biases permuted "
             "(gains halved), random = new draws from the teacher's initial distribution. Left out: did not finish, or "
             "ended with a higher task loss than at the start. Errors are relative to the shuffled baseline unless "
             "noted. Solved: task loss within 2% of the teacher's (text) or at most twice it (cycling). Output vs "
             "teacher: readout MSE (cycling), KL divergence in bits/token (text). Control: the student started from "
             "the teacher's own gains and biases.", "",
             "| task | start | students | left out | solved | task loss (teacher) | output vs teacher | activity "
             "error, before -> after | 1 - mean abs r (shuffled) | activity error in the readout's directions | ... in "
             "the other directions | gain error | bias error | control: activity error after |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|", *student_rows, ""]
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(text)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    if curves:
        training_curves(curves, os.path.join(args.out, "training_curves.png"))
        outcomes(curves, os.path.join(args.out, "outcomes.png"))
        activity_by_task(curves, os.path.join(args.out, "activity_by_task.png"))


if __name__ == "__main__":
    main()
