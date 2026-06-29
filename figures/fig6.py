# -*- coding: utf-8 -*-
"""Figure 6 (manuscript): qubit x depth landscape. -> figures/config_performance_landscape.pdf"""
from figlib import *


def build():
    T = aggT(1000); Qs = [1, 2, 4]; Ls = [2, 4, 6, 8]; cells = {}
    for r in T.values():
        cells.setdefault((Qs.index(int(r["qubits"])), Ls.index(int(r["layers"]))), []).append(fn(r["ba_mean"]))
    best = np.full((3, 4), np.nan); spread = np.full((3, 4), np.nan)
    for (qi, li), v in cells.items(): best[qi, li] = max(v); spread[qi, li] = max(v) - min(v)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, data, lab, cmap in [(axes[0], best, "(a) Best balanced accuracy", "viridis"), (axes[1], spread, "(b) Sensitivity (max $-$ min)", "magma")]:
        im = ax.imshow(data, cmap=cmap, aspect="auto", origin="lower")
        ax.set_xticks(range(4)); ax.set_xticklabels(Ls); ax.set_yticks(range(3)); ax.set_yticklabels(Qs)
        ax.set_xlabel("Layers $L$"); ax.set_ylabel("Qubits"); plab(ax, lab)
        rng = np.nanmax(data) - np.nanmin(data) + 1e-9
        for i in range(3):
            for j in range(4):
                if not np.isnan(data[i, j]):
                    nv = (data[i, j] - np.nanmin(data)) / rng
                    ax.text(j, i, f"{data[i,j]:.3f}", ha="center", va="center", fontsize=18,
                            fontweight="bold", color="white" if nv < 0.45 else "black")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for t in cb.ax.get_yticklabels(): t.set_fontweight("bold"); t.set_fontsize(16)
        bold(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "config_performance_landscape.pdf"), bbox_inches="tight"); plt.close(fig); print("Fig6 saved")


if __name__ == "__main__":
    build()
