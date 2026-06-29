# -*- coding: utf-8 -*-
"""Figure 3 (manuscript): training-size robustness. -> figures/Figure-3.pdf"""
from figlib import *


def top2(T):
    a = aggT(T); r = sorted(a.values(), key=lambda x: -fn(x["ba_mean"])); return r[0]["config"], r[1]["config"]


def pfv(T, c):
    return {(r["seed"], r["fold"]): fn(r["ba"]) for r in perf if r["train"] == str(T) and r["config"] == c}


def build():
    sizes = [100, 200, 500, 1000]
    fig, (a, b) = plt.subplots(1, 2, figsize=(17, 6.2))
    bd, pos, cols, labels, gaps = [], [], [], [], []
    for si, T in enumerate(sizes):
        c1, c2 = top2(T); d1, d2 = pfv(T, c1), pfv(T, c2)
        base = si * 3; bd += [list(d1.values()), list(d2.values())]
        pos += [base + 0.6, base + 1.4]; cols += [BLU, ORG]; labels.append((base + 1.0, f"T={T}"))
        k = sorted(set(d1) & set(d2)); gaps.append([d1[x] - d2[x] for x in k])
    bp = a.boxplot(bd, positions=pos, widths=0.6, patch_artist=True,
                   flierprops=dict(marker="o", ms=5, mfc="none", mec="#777"))
    for p, c in zip(bp["boxes"], cols): p.set_facecolor(c); p.set_alpha(0.6); p.set_linewidth(1.6)
    for m in bp["medians"]: m.set_color("black"); m.set_linewidth(2.2)
    a.set_xticks([p for p, _ in labels]); a.set_xticklabels([l for _, l in labels])
    a.set_ylabel("Seed-by-fold balanced accuracy"); plab(a, "(a) Top-2 accuracy by training size"); a.grid(axis="y", alpha=0.25)
    a.legend(handles=[Patch(fc=BLU, alpha=0.6, label="rank 1"), Patch(fc=ORG, alpha=0.6, label="rank 2")],
             loc="lower right", frameon=False)
    np.random.seed(0)
    for si, g in enumerate(gaps):
        x = np.full(len(g), si) + np.random.uniform(-0.08, 0.08, len(g))
        b.scatter(x, g, c="#888", s=45, zorder=2)
        b.errorbar(si, np.mean(g), yerr=np.std(g, ddof=1), fmt="o", color=RED, ms=13, capsize=6, zorder=3, lw=2.5)
    b.axhline(0, ls="--", color="#888", lw=1.5); b.set_xticks(range(4)); b.set_xticklabels([f"T={t}" for t in sizes])
    b.set_ylabel("Rank1 $-$ rank2 balanced accuracy", fontsize=15); plab(b, "(b) Top-2 gap by training size"); b.grid(axis="y", alpha=0.25)
    bold(a); bold(b); fig.tight_layout()
    fig.savefig(os.path.join(FIG, "Figure-3.pdf"), bbox_inches="tight"); plt.close(fig); print("Fig3 saved")


if __name__ == "__main__":
    build()
