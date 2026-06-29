# -*- coding: utf-8 -*-
"""Figure 4 (manuscript): ranking + recall/specificity + scaling.
-> figures/top_15_and_recall_specificity_combo_zoomed_top2_red.pdf"""
from figlib import *


def build():
    T = aggT(1000); ranked = sorted(T.values(), key=lambda r: -fn(r["ba_mean"]))
    top15 = ranked[:15]; rankpos = {r["config"]: i for i, r in enumerate(ranked)}
    fig, axs = plt.subplots(1, 3, figsize=(24, 7)); a, b, c = axs
    # (a) top-15 ranking
    names = [r["config"] for r in top15][::-1]; bas = [fn(r["ba_mean"]) for r in top15][::-1]
    sds = [fn(r["ba_sd"]) for r in top15][::-1]; cols = ([RED, ORG, PUR] + [BLU] * 12)[::-1]
    a.barh(range(15), bas, xerr=sds, color=cols, ecolor="#444", capsize=3, height=0.7, lw=1.4)
    a.set_yticks(range(15)); a.set_yticklabels(names, fontsize=16)
    a.set_xlim(0.80, 0.87); a.set_xlabel("Mean balanced accuracy"); plab(a, "(a) Top-15 ranking"); a.grid(axis="x", alpha=0.25)
    # (b) recall vs specificity (non-top -> BLACK)
    for r in T.values():
        i = rankpos[r["config"]]; rec = fn(r["rec_mean"]); sp = fn(r["spec_mean"])
        if i == 0: cc, z, s = RED, 5, 200
        elif i == 1: cc, z, s = ORG, 5, 200
        elif i == 2: cc, z, s = PUR, 5, 200
        elif i < 15: cc, z, s = BLU, 4, 120
        else: cc, z, s = BLK, 2, 70
        b.scatter(rec, sp, c=cc, s=s, zorder=z, edgecolors="none")
    b.set_xlabel("Mean recall (malicious)"); b.set_ylabel("Mean specificity (benign)"); plab(b, "(b) Recall vs specificity"); b.grid(alpha=0.25)
    b.legend(handles=[Line2D([0], [0], marker="o", color="w", mfc=RED, ms=14, label="rank 1"),
                      Line2D([0], [0], marker="o", color="w", mfc=ORG, ms=14, label="rank 2"),
                      Line2D([0], [0], marker="o", color="w", mfc=PUR, ms=14, label="rank 3"),
                      Line2D([0], [0], marker="o", color="w", mfc=BLU, ms=12, label="ranks 4-15"),
                      Line2D([0], [0], marker="o", color="w", mfc=BLK, ms=11, label="ranks 16-72")],
             loc="lower left", frameon=False, fontsize=17)
    # (c) mean BA vs layers, one line per qubit
    Ls = [2, 4, 6, 8]
    for q, col, mk in [(1, GRN, "o"), (2, BLU, "s"), (4, RED, "^")]:
        ys, es = [], []
        for L in Ls:
            v = [fn(r["ba_mean"]) for r in T.values() if int(r["qubits"]) == q and int(r["layers"]) == L]
            ys.append(np.mean(v)); es.append(np.std(v, ddof=1) if len(v) > 1 else 0)
        c.errorbar(Ls, ys, yerr=es, marker=mk, color=col, ms=13, capsize=6, lw=3, label=f"$Q={q}$")
    c.set_xticks(Ls); c.set_xlabel("Layers $L$ (depth)"); c.set_ylabel("Mean balanced accuracy")
    plab(c, "(c) Scaling with depth"); c.grid(alpha=0.25); c.legend(frameon=False)
    for ax in axs: bold(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "top_15_and_recall_specificity_combo_zoomed_top2_red.pdf"), bbox_inches="tight")
    plt.close(fig); print("Fig4 saved")


if __name__ == "__main__":
    build()
