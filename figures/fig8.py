# -*- coding: utf-8 -*-
"""Figure 8 (manuscript): QRGS vs classical/Fourier baselines (BA + ROC + PR).
-> figures/classical_compare_no_xgb_figure.pdf"""
from figlib import *


def build():
    cl = {r["model"]: r for r in cls if r["train"] == "1000"}; qb = A1000["M2P-Q4-L8-E-SR"]
    order = [("Fourier (K=4)", cl["Fourier_K4"]), ("MLP", cl["MLP"]), ("Fourier (K=2)", cl["Fourier_K2"]),
             ("RBF SVM", cl["RBF_SVM"]), ("QRGS best", qb), ("Logistic", cl["LogReg"]), ("Linear SVC", cl["LinearSVC"])]
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(24, 7))
    names = [o[0] for o in order]; bas = [fn(o[1]["ba_mean"]) for o in order]; sds = [fn(o[1]["ba_sd"]) for o in order]
    colb = [RED if n == "QRGS best" else "#555" for n in names]
    a.bar(range(len(names)), bas, yerr=sds, color=colb, ecolor="#888", capsize=4, width=0.66)
    a.set_xticks(range(len(names))); a.set_xticklabels(names, rotation=30, ha="right", fontsize=17)
    a.set_ylabel("Mean balanced accuracy"); a.set_ylim(0.78, 0.93); plab(a, "(a) Balanced accuracy"); a.grid(axis="y", alpha=0.25)
    for xi, v in zip(range(len(names)), bas): a.text(xi, v + sds[xi] + 0.004, f"{v:.3f}", ha="center", fontsize=14, fontweight="bold")
    PAL = {"Fourier K4": "#1b9e77", "MLP": "#7570b3", "RBF SVM": "#1f77b4", "QRGS best": RED, "Logistic": "#888888"}
    ORD = ["Fourier K4", "MLP", "RBF SVM", "QRGS best", "Logistic"]
    bym = {}
    for r in roc:
        d = bym.setdefault(r["model"], {"fpr": [], "tpr": [], "auc": r["auc"]}); d["fpr"].append(fn(r["fpr"])); d["tpr"].append(fn(r["tpr"]))
    for m in ORD:
        if m in bym: b.plot(bym[m]["fpr"], bym[m]["tpr"], color=PAL[m], lw=3.2, label=f"{m} ({bym[m]['auc']})")
    b.plot([0, 1], [0, 1], ls=":", color="#bbb", lw=2)
    b.set_xlabel("False positive rate"); b.set_ylabel("True positive rate"); plab(b, "(b) ROC")
    b.legend(fontsize=16, frameon=False, loc="lower right", title="AUC", title_fontsize=16); b.grid(alpha=0.25)
    byp = {}
    for r in pr:
        d = byp.setdefault(r["model"], {"rec": [], "prec": [], "ap": r["ap"]}); d["rec"].append(fn(r["recall"])); d["prec"].append(fn(r["precision"]))
    for m in ORD:
        if m in byp: c.plot(byp[m]["rec"], byp[m]["prec"], color=PAL[m], lw=3.2, label=f"{m} ({byp[m]['ap']})")
    c.set_xlabel("Recall"); c.set_ylabel("Precision"); plab(c, "(c) Precision-recall")
    c.legend(fontsize=16, frameon=False, loc="lower center", title="AP", title_fontsize=16); c.grid(alpha=0.25)
    bold(a); bold(b); bold(c); fig.tight_layout()
    fig.savefig(os.path.join(FIG, "classical_compare_no_xgb_figure.pdf"), bbox_inches="tight"); plt.close(fig); print("Fig8 saved")


if __name__ == "__main__":
    build()
