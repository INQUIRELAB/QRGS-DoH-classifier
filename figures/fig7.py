# -*- coding: utf-8 -*-
"""Figure 7 (manuscript): factorial (marginal) effects of each design axis.
-> figures/hyperparameter_influence_4panel.pdf"""
from figlib import *


def build():
    T = aggT(1000); rows = list(T.values())
    axes_def = [("(a) Scaler", ["M01", "M2P"], lambda r: r["scaler"]),
                ("(b) Qubits", ["1", "2", "4"], lambda r: r["qubits"]),
                ("(c) Layers", ["2", "4", "6", "8"], lambda r: r["layers"]),
                ("(d) Entangle", ["NE", "E"], lambda r: r["ent"]),
                ("(e) Reupload", ["SR", "AR"], lambda r: r["method"])]
    gm = np.mean([fn(r["ba_mean"]) for r in rows])
    fig, axes = plt.subplots(1, 5, figsize=(24, 6), sharey=True)
    for ax, (name, levels, key) in zip(axes, axes_def):
        m = [np.mean([fn(r["ba_mean"]) for r in rows if key(r) == lv]) for lv in levels]
        s = [np.std([fn(r["ba_mean"]) for r in rows if key(r) == lv], ddof=1) for lv in levels]
        ax.axhline(gm, ls="--", color="#888", lw=1.5, zorder=0)
        ax.errorbar(range(len(levels)), m, yerr=s, fmt="o-", color=BLU, ms=13, capsize=6, lw=3)
        delta = max(m) - min(m)
        ax.text(0.04, 0.96, f"$\\Delta$={delta:.3f}", transform=ax.transAxes,
                ha="left", va="top", fontsize=20, fontweight="bold")
        ax.set_xticks(range(len(levels))); ax.set_xticklabels(levels)
        ax.set_title(name, loc="left", fontsize=22, fontweight="bold"); ax.grid(alpha=0.25); bold(ax)
    axes[0].set_ylabel("Mean balanced accuracy")
    y0, y1 = axes[0].get_ylim()
    axes[0].set_ylim(y0, y1 + 0.14 * (y1 - y0))  # headroom so error caps clear the Delta text
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "hyperparameter_influence_4panel.pdf"), bbox_inches="tight"); plt.close(fig); print("Fig7 saved")


if __name__ == "__main__":
    build()
