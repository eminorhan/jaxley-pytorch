"""Stress test of the cycling result: can a student reach both the teacher's task loss and
the teacher's single-neuron activity? Compares training strategies (see README.md).

Reads `<runs>/<arm>/seed<k>/result.json` for every strategy in STRATEGIES (the arms of
configs/cycling_*.toml) and the control (`cycling_control`), and writes to `--out`:

  stress_test.md   per strategy: students, how many ran away (left out: non-finite task
                   loss or a higher one than at the start), how many the time limit stopped
                   (their last evaluation counts), task loss and activity error (median,
                   best), and how many reach the target (task loss at most
                   twice the teacher's and activity error at most 0.1 of the shuffled
                   baseline);
  stress_test.png  one panel per strategy: final task loss against final activity error,
                   the strategy's students in blue (for an L-BFGS finish: a line from where
                   Adam left them), the other strategies' in gray, the control in aqua.

Usage: python stress_test.py --runs <runs dir> --out <dir>
"""
import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from jaxley_pytorch.training.report import COLORS, INK, INK2, MUTED, SURFACE, load, style


def ran_away(s):
    """Ended with a non-finite task loss or a higher one than at the start. (A run stopped
    by the time limit keeps its last evaluation.)"""
    first, last = s["evals"][0]["task_loss"], s["evals"][-1]["task_loss"]
    return not np.isfinite(last) or last > first

STRATEGIES = [  # (arm: experiment_name of its config, label)
    ("cycling_students", "shuffled start (the paper's)"),
    ("cycling_shuffled_lbfgs", "shuffled start + L-BFGS"),
    ("cycling_students_random", "random start"),
    ("cycling_rstable_lbfgs", "random start, stable draws + L-BFGS"),
    ("cycling_uniform_lbfgs", "uniform start + L-BFGS"),
    ("cycling_rstable_fresh", "random start, stable draws, 6,400 trials"),
    ("cycling_rstable_fresh_long", "... 30k steps, batch 128, warmup 1k"),
]
TARGET_LOSS, TARGET_ACT = 2.0, 0.1  # x the teacher's task loss; x the shuffled baseline
CONTEXT = "#c3c2b7"


def points(s, base):
    """(task loss, activity error / baseline) at the end of Adam and at the end."""
    adam = [e for e in s["evals"] if "phase" not in e][-1]
    last = s["evals"][-1]
    p = lambda e: (e["task_loss"], e["act_mse"] / base["act_mse_shuffled"])
    return p(adam), p(last)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    control = load(os.path.join(args.runs, "cycling_control", "seed0", "result.json"))
    base = control["baselines"]
    teacher_loss = control["evals"][0]["teacher_task_loss"]

    groups = []
    for arm, label in STRATEGIES:
        dirs = sorted(glob.glob(os.path.join(args.runs, arm, "seed*")))
        if not dirs:
            continue
        results = [load(os.path.join(d, "result.json")) for d in dirs if os.path.exists(os.path.join(d, "result.json"))]
        kept = [s for s in results if not ran_away(s)]
        n_cut = sum("final" not in s for s in kept)
        groups.append((label, [points(s, base) for s in kept], len(dirs) - len(kept), n_cut))

    rows = ["| strategy | students | ran away (left out) | stopped by the time limit | task loss: median, best | "
            "activity error / baseline: median, best | in the target |", "|---|---|---|---|---|---|---|"]
    for label, pts, n_out, n_cut in groups:
        loss = np.array([p[1][0] for p in pts])
        act = np.array([p[1][1] for p in pts])
        hit = int(np.sum((loss <= TARGET_LOSS * teacher_loss) & (act <= TARGET_ACT)))
        rows.append(f"| {label} | {len(pts) + n_out} | {n_out} | {n_cut} | {np.median(loss):.3g}, {loss.min():.3g} | "
                    f"{np.median(act):.3g}, {act.min():.3g} | {hit} |")
    text = "\n".join([f"Teacher task loss {teacher_loss:.3g}; target: task loss <= {TARGET_LOSS:g}x the teacher's and "
                      f"activity error <= {TARGET_ACT:g} of the shuffled baseline. Control (teacher's own start): task "
                      f"loss {control['evals'][-1]['task_loss']:.3g}, activity error "
                      f"{control['evals'][-1]['act_mse'] / base['act_mse_shuffled']:.3g}.", ""] + rows + [""])
    print(text)
    with open(os.path.join(args.out, "stress_test.md"), "w") as f:
        f.write(text)

    c_end = control["evals"][-1]
    every = [q for _, pts, _, _ in groups for p in pts for q in p] + [(c_end["task_loss"], c_end["act_mse"] / base[
        "act_mse_shuffled"]), (teacher_loss, TARGET_ACT)]
    xs, ys = [p[0] for p in every], [p[1] for p in every]
    xlim = (10 ** (np.log10(min(xs)) - 0.3), 10 ** (np.log10(max(xs)) + 0.3))
    ylim = (10 ** (np.log10(min(ys)) - 0.3), 10 ** (np.log10(max(ys)) + 0.3))
    ncol = 3
    nrow = int(np.ceil(len(groups) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 4.3 * nrow + 0.5), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for k, (label, pts, n_out, n_cut) in enumerate(groups):
        ax = axes[k // ncol, k % ncol]
        style(ax)
        for j, (_, others, _, _) in enumerate(groups):
            if j != k:
                ax.scatter([p[1][0] for p in others], [p[1][1] for p in others], s=18, color=CONTEXT, zorder=2,
                           linewidth=0)
        for adam, last in pts:
            if adam != last:
                ax.plot([adam[0], last[0]], [adam[1], last[1]], color=COLORS["shuffled"], lw=1.0, alpha=0.6, zorder=3)
                ax.scatter([adam[0]], [adam[1]], s=22, facecolor=SURFACE, edgecolor=COLORS["shuffled"], linewidth=1.2,
                           zorder=3)
        ax.scatter([p[1][0] for p in pts], [p[1][1] for p in pts], s=42, color=COLORS["shuffled"],
                   edgecolor=SURFACE, linewidth=1.5, zorder=4)
        ax.scatter([c_end["task_loss"]], [c_end["act_mse"] / base["act_mse_shuffled"]], s=50, marker="D",
                   color=COLORS["control"], edgecolor=SURFACE, linewidth=1.5, zorder=4)
        ax.axhline(1.0, color=MUTED, lw=1.0, zorder=1)
        ax.axvline(teacher_loss, color=INK, lw=1.0, zorder=1)
        ax.plot([xlim[0], TARGET_LOSS * teacher_loss, TARGET_LOSS * teacher_loss], [TARGET_ACT, TARGET_ACT, ylim[0]],
                color=INK2, lw=1.0, zorder=1)
        ax.annotate("target", (TARGET_LOSS * teacher_loss, TARGET_ACT), xytext=(3, 3), textcoords="offset points",
                    fontsize=7.5, color=INK2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(label, fontsize=9, color=INK, loc="left")
        notes = ([f"{n_out} ran away, not shown"] if n_out else []) + (
            [f"{n_cut} hit the time limit"] if n_cut else [])
        if notes:
            ax.text(0.99, 0.02, "\n".join(notes), transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
                    color=INK2)
        if k // ncol == nrow - 1:
            ax.set_xlabel("final task loss (MSE)", fontsize=8)
        if k % ncol == 0:
            ax.set_ylabel("activity error / shuffled-neuron baseline", fontsize=8)
    for k in range(len(groups), nrow * ncol):
        axes[k // ncol, k % ncol].axis("off")
    handles = [plt.Line2D([], [], ls="", marker="o", ms=7, color=COLORS["shuffled"]),
               plt.Line2D([], [], ls="", marker="o", ms=6, markerfacecolor=SURFACE, color=COLORS["shuffled"]),
               plt.Line2D([], [], ls="", marker="o", ms=5, color=CONTEXT),
               plt.Line2D([], [], ls="", marker="D", ms=6, color=COLORS["control"]),
               plt.Line2D([], [], color=INK, lw=1.0), plt.Line2D([], [], color=MUTED, lw=1.0)]
    labels = ["this strategy's students", "... where Adam left them (before L-BFGS)", "other strategies",
              "control: teacher's own start", "teacher's task loss", "shuffled neurons"]
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, fontsize=8.5, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / (4.3 * nrow + 0.5)))
    fig.savefig(os.path.join(args.out, "stress_test.png"), dpi=150, facecolor=SURFACE)


if __name__ == "__main__":
    main()
