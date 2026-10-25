"""Tables and figure of the task-complexity experiment (see README.md).

Reads `<runs>/<task>/teacher/result.json`, `<runs>/<task>/student_seed*/result.json` and
`<runs>/<task>/student_control/result.json` (tasks cycling, chars, tokens; missing ones
are skipped) and writes `summary.md`, `summary.json` and `fig1.png` (the paper's
Fig. 1c-e for every task, the control student in black) to `--out`.

Usage: python summarize.py --runs <runs dir> --out <dir>
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


def load(path):
    with open(path) as f:
        return json.load(f)


def spread(values, fmt="{:.3g}"):
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return "-"
    return (fmt + " [" + fmt + ", " + fmt + "]").format(np.median(v), v.min(), v.max())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    summary, teacher_rows, student_rows, curves = {}, [], [], {}
    for task in TASKS:
        tpath = os.path.join(args.runs, task, "teacher", "result.json")
        spaths = sorted(glob.glob(os.path.join(args.runs, task, "student_seed*", "result.json")))
        cpath = os.path.join(args.runs, task, "student_control", "result.json")
        if not os.path.exists(tpath) and not spaths:
            continue
        entry = {}
        if os.path.exists(tpath):
            t = load(tpath)["evals"][-1]
            entry["teacher"] = t
            loss = (f"MSE {t['task_loss']:.3g}" if task == "cycling" else
                    f"{t['task_loss']:.3f} bits/token, {t['val_bits_per_char']:.3f} bits/char")
            teacher_rows.append(f"| {task} | {loss} | {t['participation_ratio']:.1f} | {t['components_90']} | "
                                f"{t['readout_rank']} | {t['readout_variance_fraction']:.3f} | {t['mean_rate']:.3g} |")
        students = [load(p) for p in spaths]
        control = load(cpath) if os.path.exists(cpath) else None
        if students:
            base = students[0]["baselines"]
            first = [s["evals"][0] for s in students]
            last = [s["evals"][-1] for s in students]
            ratio = lambda evs, key: [e[key] / base[f"{key}_shuffled"] for e in evs]
            solved = sum(e["task_loss"] <= SOLVED[task] * e["teacher_task_loss"] for e in last)
            control_ratio = (f"{control['evals'][-1]['act_mse'] / base['act_mse_shuffled']:.3f}"
                             if control else "-")
            entry["students"] = {"n": len(students), "solved": solved, "baselines": base, "initial": first,
                                 "final": last, "act_ratio_initial": ratio(first, "act_mse"),
                                 "act_ratio_final": ratio(last, "act_mse"),
                                 "control_final": control["evals"][-1] if control else None}
            student_rows.append(
                f"| {task} | {len(students)} | {solved} | {spread([e['task_loss'] for e in last])} "
                f"({last[0]['teacher_task_loss']:.3g}) | {spread([e['output_vs_teacher'] for e in last])} | "
                f"{spread(ratio(first, 'act_mse'), '{:.2f}')} -> {spread(ratio(last, 'act_mse'), '{:.2f}')} | "
                f"{spread([e['act_corr_err'] for e in last], '{:.3f}')} ({base['act_corr_err_shuffled']:.3f}) | "
                f"{spread(ratio(last, 'act_mse_readout'), '{:.2f}')} | {spread(ratio(last, 'act_mse_other'), '{:.2f}')} | "
                f"{spread(ratio(last, 'gain_rmse'), '{:.2f}')} | {spread(ratio(last, 'bias_rmse'), '{:.2f}')} | "
                f"{control_ratio} |")
            curves[task] = (students, control, base)
        summary[task] = entry

    lines = ["## Teachers", "",
             "| task | task loss (validation) | participation ratio | components for 90% | readout rank | "
             "variance in the readout's directions | mean rate |",
             "|---|---|---|---|---|---|---|", *teacher_rows, "",
             "## Students", "",
             "Median [min, max] over the students started from shuffled gains and biases. Errors are relative to "
             "the shuffled baseline unless noted. Solved: task loss within 2% of the teacher's (text) or at most twice "
             "it (cycling). Output vs teacher: readout MSE (cycling), KL divergence in bits/token (text). Control: the "
             "student started from the teacher's own gains and biases.", "",
             "| task | students | solved | task loss (teacher) | output vs teacher | activity error, before -> after "
             "| 1 - mean abs r (shuffled) | activity error in the readout's directions | ... in the other directions "
             "| gain error | bias error | control: activity error after |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|", *student_rows, ""]
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(text)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    if not curves:
        return
    cols = [("task_loss", None), ("act_mse", "act_mse_shuffled"), ("gain_rmse", "gain_rmse_shuffled"),
            ("bias_rmse", "bias_rmse_shuffled")]
    titles = [None, "error in activity", "error in gains", "error in biases"]
    fig, axes = plt.subplots(len(curves), 4, figsize=(15, 3.2 * len(curves)), squeeze=False)
    for row, (task, (students, control, base)) in enumerate(curves.items()):
        colors = plt.cm.Oranges(np.linspace(0.35, 0.9, len(students)))
        runs = [(s, c, 1.0) for s, c in zip(students, colors)] + ([(control, "k", 1.2)] if control else [])
        for col, (key, base_key) in enumerate(cols):
            ax = axes[row, col]
            for s, c, lw in runs:
                ax.plot([e["step"] for e in s["evals"]], [e[key] for e in s["evals"]], color=c, lw=lw)
            if base_key is not None:
                ax.axhline(base[base_key], color="0.6", lw=2, label="shuffled neurons")
            else:
                ax.axhline(students[0]["evals"][0]["teacher_task_loss"], color="0.6", lw=2, ls="--", label="teacher")
            if col < 2:
                ax.set_yscale("log")
            ax.set_xlabel("training step")
            ax.set_title(f"{task}: {titles[col] or LOSS[task]}", fontsize=9)
            if row == 0 and col < 2:
                ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig1.png"), dpi=130)


if __name__ == "__main__":
    main()
