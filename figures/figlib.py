# -*- coding: utf-8 -*-
"""Shared style + data + helpers for the per-figure result scripts (fig3..fig8).
npj style: large BOLD fonts, panel letters. Axis-label size reduced from 28 to 22
so long y-labels (e.g. 'Rank1 - rank2 balanced accuracy') no longer dominate."""
import csv, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": "Times New Roman", "mathtext.fontset": "stix",
    "mathtext.default": "bf", "pdf.fonttype": 42, "svg.fonttype": "none",
    "font.size": 24, "font.weight": "bold",
    "axes.labelweight": "bold", "axes.titleweight": "bold",
    "axes.labelsize": 22, "axes.titlesize": 24,
    "xtick.labelsize": 20, "ytick.labelsize": 20, "legend.fontsize": 18,
    "axes.linewidth": 1.4, "lines.linewidth": 3.0, "lines.markersize": 11,
    "xtick.major.width": 1.4, "ytick.major.width": 1.4,
})

A = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(A, "..", "figures")


def rd(n):
    with open(os.path.join(A, n)) as f:
        return list(csv.DictReader(f))


agg = rd("qrgs_rerun_aggregated.csv"); cls = rd("classical_fc_aggregated.csv")
perf = rd("perfold_fc.csv"); conv = rd("convergence_fc.csv")
convb = rd("convergence_budget.csv"); roc = rd("roc_curves.csv"); pr = rd("pr_curves.csv")


def fn(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def aggT(T):
    return {r["config"]: r for r in agg if r["train"] == str(T)}


A1000 = aggT(1000)
RED, ORG, PUR, BLU, BLK, GRN = "#d62728", "#ff7f0e", "#9467bd", "#1f77b4", "#000000", "#2ca02c"


def bold(ax):
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")


def plab(ax, s, fs=22):
    """Panel title (letter + optional description), left-aligned above the axes."""
    ax.set_title(s, loc="left", fontsize=fs, fontweight="bold", pad=8)
