# This file is part of jaxley-pytorch, a PyTorch port of Jaxley, a differentiable
# neuroscience simulator. Jaxley is licensed under the Apache License Version 2.0, see
# <https://www.apache.org/licenses/>
"""What the experiments' summary scripts share: reading runs' `result.json`, the rule
for leaving a run out, and the style of the figures (the dataviz reference palette:
categorical slots 1-3 and the chart ink)."""

import json

import matplotlib
import matplotlib.ticker
import numpy as np

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
# Students by how they started (init.STARTS: shuffled; random; the teacher's own).
COLORS = {"shuffled": "#2a78d6", "random": "#eb6834", "control": "#1baf7a"}


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def excluded(result: dict) -> bool:
    """Did not finish, or ended with a higher task loss than it started with (its
    activity ran away)."""
    first, last = result["evals"][0]["task_loss"], result["evals"][-1]["task_loss"]
    return "final" not in result or not np.isfinite(last) or last > first


def neurons(result: dict) -> int:
    """The number of neurons of a run (also of runs made before this library)."""
    return result["neurons"] if "neurons" in result else result["args"]["n"]


def spread(values, fmt: str = "{:.3g}") -> str:
    """The median and the range of `values`, as "median [min, max]"."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return "-"
    return (fmt + " [" + fmt + ", " + fmt + "]").format(np.median(v), v.min(), v.max())


def steps_in_thousands(ax):
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5, integer=True))
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1000:g}k" if x else "0")
    )


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
