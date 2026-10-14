"""Figure for the README: training throughput vs network size (left) and validation
loss vs characters trained (right).

Usage: python plot_results.py <bench dir> <bench dir for the spiking models> <run dir>,<run dir>,... <out.png>
"""
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical slots 1-5 of the reference palette (fixed order), one hue per model.
COLORS = {"gru": "#2a78d6", "rate": "#eb6834", "hh": "#1baf7a", "pospischil": "#eda100", "l5pc": "#e87ba4"}
NAMES = {"gru": "GRU (dense)", "rate": "rate", "hh": "Hodgkin-Huxley", "pospischil": "Pospischil RS/FS",
         "l5pc": "L5PC (3 compartments)"}
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def fastest(dirs, models):
    runs = {}
    for root in dirs:
        for path in glob.glob(os.path.join(root, "*", "bench.json")):
            r = json.load(open(path))
            if r["model"] not in models:
                continue
            key = (r["model"], r["n"])
            if key not in runs or r["tokens_per_s"] > runs[key]["tokens_per_s"]:
                runs[key] = r
    return runs


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=1.0, linestyle="-")
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT2, labelsize=9)
    ax.xaxis.label.set_color(TEXT2)
    ax.yaxis.label.set_color(TEXT2)


def main():
    bench_dir, bench_spiking_dir, run_dirs, out = sys.argv[1:5]
    runs = fastest([bench_dir], {"gru", "rate"})
    runs.update(fastest([bench_spiking_dir], {"hh", "pospischil", "l5pc"}))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    for model in COLORS:
        pts = sorted((n, r["tokens_per_s"]) for (m, n), r in runs.items() if m == model)
        if not pts:
            continue
        ns, tps = zip(*pts)
        ax1.plot(ns, tps, color=COLORS[model], linewidth=2, solid_capstyle="round", label=NAMES[model])
        ax1.plot(ns, tps, "o", color=COLORS[model], markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xlabel("neurons (100 synapses each; GRU: units, dense)")
    ax1.set_ylabel("training throughput (characters / s)")
    ax1.set_title("Throughput on one H100 (float32; batch 32, the largest batch 8)", color=TEXT, fontsize=11, loc="left")
    style(ax1)
    ax1.legend(frameon=False, fontsize=8.5, labelcolor=TEXT2, loc="upper right")

    for run in run_dirs.split(","):
        r = json.load(open(os.path.join(run, "result.json")))
        log = r["log"]
        if not log["val_bpc"]:
            continue
        chars = [s * r["batch"] * r["chunk"] for s in log["val_step"]]
        big = r["n"] > 1024 or (r["model"] == "gru" and r["n"] < 1024)
        label = f"{NAMES[r['model']]}, {r['n']:,}" + (f" (lr {r['args']['lr']:g})" if r["model"] == "gru" else "")
        ax2.plot(chars, log["val_bpc"], color=COLORS[r["model"]], linewidth=2, linestyle="--" if big else "-",
                 solid_capstyle="round", label=label)
        ax2.plot(chars[-1:], log["val_bpc"][-1:], "o", color=COLORS[r["model"]], markersize=6,
                 markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax2.set_xscale("log")
    ax2.set_xlabel("characters trained")
    ax2.set_ylabel("validation loss (bits per character)")
    ax2.set_title("Training (4 h per biophysical network, 1 h per GRU)", color=TEXT, fontsize=11, loc="left")
    style(ax2)
    ax2.legend(frameon=False, fontsize=8.5, labelcolor=TEXT2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("saved", out)


if __name__ == "__main__":
    main()
