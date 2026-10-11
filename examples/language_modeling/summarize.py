"""Collect `train.py --mode bench` and `--mode train` results into markdown tables.

Usage:
  python summarize.py bench <dir>[,<dir>...] [<model>,<model>...]
  python summarize.py train <dir>
  python summarize.py matched <dir>     # training loss at equal numbers of characters
"""
import glob
import json
import math
import os
import sys

ORDER = ["gru", "rate", "hh", "pospischil", "l5pc"]


def bench_table(roots, models=None):
    runs = {}
    for root in roots:
        for path in glob.glob(os.path.join(root, "*", "bench.json")):
            r = json.load(open(path))
            if models and r["model"] not in models:
                continue
            key = (r["model"], r["n"])
            # The fastest configuration that worked for every (model, n).
            if key not in runs or r["tokens_per_s"] > runs[key]["tokens_per_s"]:
                runs[key] = r
    lines = ["| model | neurons | synapses | solver step (ms) | batch | ms per solver step | "
             "characters / s | days per 100M characters | peak GPU memory (GB) | GPU |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for (model, n), r in sorted(runs.items(), key=lambda kv: (ORDER.index(kv[0][0]), kv[0][1])):
        ms = r.get("ms_per_solver_step")
        syn = "-" if model == "gru" else f"{r['synapses']:,}"
        lines.append(
            f"| {model} | {n:,} | {syn} | {'-' if model == 'gru' else r['dt_ms']} | {r['batch']} | "
            f"{'-' if ms is None else f'{ms:.2f}'} | {r['tokens_per_s']:,.0f} | {1e8 / r['tokens_per_s'] / 86400:.3g} | "
            f"{r['peak_memory_gb']:.1f} | {r['gpu'].replace('NVIDIA ', '').replace(' 80GB HBM3', '')} |")
    return "\n".join(lines)


def train_table(root):
    lines = ["| model | neurons | learning rate | parameters | characters trained | hours | characters / s | "
             "train bpc (last 100 steps) | validation bpc |",
             "|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for path in glob.glob(os.path.join(root, "*", "result.json")):
        r = json.load(open(path))
        log = r["log"]
        chars = r["steps"] * r["batch"] * r["chunk"]
        val = log["val_bpc"][-1] if log["val_bpc"] else float("nan")
        n_params = sum(r["params"].values())
        rows.append((ORDER.index(r["model"]), r["n"], -r["args"]["lr"], (
            f"| {r['model']} | {r['n']:,} | {r['args']['lr']:g} | {n_params / 1e6:.2f}M | {chars / 1e6:,.1f}M | "
            f"{log['elapsed_s'][-1] / 3600:.1f} | {r['tokens_per_s']:,.0f} | "
            f"{log['loss'][-1] / math.log(2):.2f} | {val:.2f} |")))
    lines += [row[-1] for row in sorted(rows)]
    return "\n".join(lines)


def matched_table(root, points=(0.5e6, 1e6, 2e6, 5e6, 50e6)):
    """Training loss (bits per character, mean of the logging window) once a run has
    seen `points` characters: what every model learns from the same amount of data."""
    head = " | ".join(f"{p / 1e6:g}M" for p in points)
    lines = [f"| model | neurons | learning rate | {head} |", "|---|---|---|" + "---|" * len(points)]
    rows = []
    for path in glob.glob(os.path.join(root, "*", "result.json")):
        r = json.load(open(path))
        chars = [s * r["batch"] * r["chunk"] for s in r["log"]["step"]]
        cells = []
        for p in points:
            i = next((i for i, c in enumerate(chars) if c >= p), None)
            cells.append("-" if i is None else f"{r['log']['loss'][i] / math.log(2):.2f}")
        rows.append((ORDER.index(r["model"]), r["n"], -r["args"]["lr"],
                     f"| {r['model']} | {r['n']:,} | {r['args']['lr']:g} | " + " | ".join(cells) + " |"))
    lines += [row[-1] for row in sorted(rows)]
    return "\n".join(lines)


if __name__ == "__main__":
    if sys.argv[1] == "bench":
        print(bench_table(sys.argv[2].split(","), set(sys.argv[3].split(",")) if len(sys.argv) > 3 else None))
    elif sys.argv[1] == "matched":
        print(matched_table(sys.argv[2]))
    else:
        print(train_table(sys.argv[2]))
