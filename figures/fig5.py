# -*- coding: utf-8 -*-
"""Figure 5 (manuscript): optimization convergence. -> figures/fig_convergence.pdf"""
from figlib import *


def build():
    fig, (a, b) = plt.subplots(1, 2, figsize=(18, 6.4))
    by = {}
    for r in conv: by.setdefault(r["config"], []).append((int(r["eval"]), fn(r["loss"])))
    order = sorted(by, key=lambda cc: int([r for r in conv if r["config"] == cc][0]["N"]))
    cmap = plt.cm.viridis(np.linspace(0, 0.88, len(order)))
    for cc, col in zip(order, cmap):
        pts = sorted(by[cc]); N = [r for r in conv if r["config"] == cc][0]["N"]
        a.plot([p[0] for p in pts], [p[1] for p in pts], color=col, lw=3, label=f"{cc} (N={N})")
    a.axvline(100, ls="--", color=RED, lw=2)
    a.text(112, a.get_ylim()[1] * 0.97, "original\nbudget=100", color=RED, fontsize=18, va="top", fontweight="bold")
    a.set_xscale("log"); a.set_xlabel("COBYLA objective evaluation")
    a.set_ylabel("Best training loss"); plab(a, "(a) Convergence by circuit size"); a.legend(fontsize=15, frameon=False); a.grid(alpha=0.25)
    # (b) under-budget penalty (loss@100 - converged loss) vs N, jittered by qubit
    rng = np.random.default_rng(1)
    for q, col, mk in [(1, GRN, "o"), (2, BLU, "s"), (4, RED, "^")]:
        xs, ys = [], []
        for r in convb:
            if int(r["config"].split("-Q")[1][0]) == q:
                xs.append(int(r["N"]) * (1 + rng.uniform(-0.03, 0.03))); ys.append(fn(r["penalty"]))
        b.scatter(xs, ys, color=col, s=110, marker=mk, label=f"$Q={q}$", alpha=0.85, edgecolors="white", linewidths=0.8)
    b.axhline(0, ls=":", color="#aaa", lw=1.5); b.set_xlabel("Trainable parameters $N$")
    b.set_ylabel("Extra loss vs. fully-trained model (at 100 steps)", fontsize=16); plab(b, "(b) Under-budget penalty"); b.legend(frameon=False); b.grid(alpha=0.25)
    bold(a); bold(b); fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_convergence.pdf"), bbox_inches="tight"); plt.close(fig); print("Fig5 saved")


if __name__ == "__main__":
    build()
