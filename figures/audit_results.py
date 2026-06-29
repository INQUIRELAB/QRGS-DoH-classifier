# -*- coding: utf-8 -*-
"""Audit every quantitative claim in the manuscript against source data."""
import numpy as np
import pandas as pd

F = r"Z:\Cyber-Code\manuscript\Final_manuscript\figures\New Figs\data"
df = pd.read_csv(F + r"\qrgs_seed_fold_rows.csv")

checks = []
def chk(name, ok, detail=""):
    checks.append((name, bool(ok), detail))

# ---- experiment scale ----
chk("4320 seed-by-fold rows", len(df) == 4320, f"rows={len(df)}")
chk("72 configs per train size",
    all(df[df.train == t].config.nunique() == 72 for t in [100, 200, 500, 1000]))
chk("seeds = {7,13,37,42,101}", sorted(df.seed.unique()) == [7, 13, 37, 42, 101])
chk("1440 config-seed runs", df.groupby(['train', 'seed', 'config']).ngroups == 1440)
chk("3 folds each", (df.groupby(['train', 'seed', 'config']).size() == 3).all())
chk("test sets all 4000 samples",
    ((df.tn + df.fp + df.fn + df.tp) == 4000).all())

# ---- parameter budget formula N = cQL ----
c = np.where(df.reupload == "SR", 12, 6)
chk("N = cQL for all rows", (df.params == c * df.qubits * df.layers).all())
chk("min budget = 24", df.params.min() == 24,
    f"configs at 24: {sorted(df[df.params==24].config.unique())}")
n384 = df[(df.train == 1000) & (df.params == 384)].config.nunique()
chk("only 4 configs reach N=384", n384 == 4, f"n={n384}")
chk("64 two-/four-qubit configs",
    df[(df.train == 1000) & (df.qubits >= 2)].config.nunique() == 64)

# ---- T=1000 ranking ----
g = (df[df.train == 1000].groupby("config")
     .agg(ba=("balanced_acc", "mean"), sd=("balanced_acc", "std"),
          rec=("recall", "mean"), spec=("specificity", "mean"),
          params=("params", "first"), qubits=("qubits", "first"),
          layers=("layers", "first"))
     .sort_values("ba", ascending=False))
top = g.head(15)
r1, r2, r3 = g.index[0], g.index[1], g.index[2]
chk("rank1 = M2P-Q1-L4-NE-SR", r1 == "M2P-Q1-L4-NE-SR", r1)
chk("rank1 BA 0.818+/-0.022",
    f"{g.ba.iloc[0]:.3f}" == "0.818" and f"{g.sd.iloc[0]:.3f}" == "0.022",
    f"{g.ba.iloc[0]:.4f} +/- {g.sd.iloc[0]:.4f}")
chk("rank1 params 48", g.params.iloc[0] == 48)
chk("rank2 = M2P-Q2-L6-E-AR 0.808+/-0.020",
    r2 == "M2P-Q2-L6-E-AR" and f"{g.ba.iloc[1]:.3f}" == "0.808"
    and f"{g.sd.iloc[1]:.3f}" == "0.020",
    f"{r2} {g.ba.iloc[1]:.4f} +/- {g.sd.iloc[1]:.4f}")
chk("rank3 = M2P-Q2-L4-E-AR 0.807+/-0.038",
    r3 == "M2P-Q2-L4-E-AR" and f"{g.ba.iloc[2]:.3f}" == "0.807"
    and f"{g.sd.iloc[2]:.3f}" == "0.038",
    f"{r3} {g.ba.iloc[2]:.4f} +/- {g.sd.iloc[2]:.4f}")
chk("top15 band 0.788..0.818",
    f"{top.ba.min():.3f}" == "0.788" and f"{top.ba.max():.3f}" == "0.818",
    f"{top.ba.min():.4f}..{top.ba.max():.4f}")
chk("winner recall 0.835 / spec 0.800",
    f"{g.rec.iloc[0]:.3f}" == "0.835" and f"{g.spec.iloc[0]:.3f}" == "0.800",
    f"rec {g.rec.iloc[0]:.4f} spec {g.spec.iloc[0]:.4f}")
chk("winner FAR ~20.0%", abs((1 - g.spec.iloc[0]) * 100 - 20.0) < 0.05,
    f"{(1-g.spec.iloc[0])*100:.2f}%")

# single-qubit family
q1 = g[g.index.str.contains("Q1")]
def fam(cfg):
    return (f"{g.loc[cfg].ba:.3f} +/- {g.loc[cfg].sd:.3f}"
            f" ({int(g.loc[cfg].params)}p)")
chk("Q1-L6 0.802+/-0.027 72p",
    f"{g.loc['M2P-Q1-L6-NE-SR'].ba:.3f}" == "0.802"
    and f"{g.loc['M2P-Q1-L6-NE-SR'].sd:.3f}" == "0.027"
    and g.loc['M2P-Q1-L6-NE-SR'].params == 72, fam('M2P-Q1-L6-NE-SR'))
chk("Q1-L2 0.799+/-0.015 24p",
    f"{g.loc['M2P-Q1-L2-NE-SR'].ba:.3f}" == "0.799"
    and f"{g.loc['M2P-Q1-L2-NE-SR'].sd:.3f}" == "0.015"
    and g.loc['M2P-Q1-L2-NE-SR'].params == 24, fam('M2P-Q1-L2-NE-SR'))

# four-qubit claims
q4 = g[g.qubits == 4]
chk("best Q4 = 0.790", f"{q4.ba.max():.3f}" == "0.790", f"{q4.ba.max():.4f}")
chk("no Q4 in top 8", not any(g.head(8).qubits == 4),
    f"top8 qubits: {list(g.head(8).qubits)}")
chk("M2P in all top 8", all(cfg.startswith("M2P") for cfg in g.head(8).index),
    f"{list(g.head(8).index)}")
sr_top8 = sum(cfg.endswith("SR") for cfg in g.head(8).index)
chk("SR frequent in top 8 (info)", True, f"SR count in top8: {sr_top8}/8")

# heatmap: best cell at (Q1, L4); four-qubit cells lower maxima
cell = g.reset_index().groupby(["qubits", "layers"]).ba.max()
best_cell = cell.idxmax()
chk("heatmap peak at Q=1,L=4", best_cell == (1, 4), str(best_cell))
chk("all Q4 cell maxima < Q1/Q2 best",
    cell.loc[4].max() < max(cell.loc[1].max(), cell.loc[2].max()),
    f"Q4 max {cell.loc[4].max():.3f}")

# ---- Table 2 (all training sizes) ----
t2 = pd.read_csv(r"Z:\Cyber-Code\manuscript\Final_manuscript\analysis"
                 r"\seed_aggregate\qrgs_top2_paired_tests.csv")
expect = {
    100: ("M01-Q1-L6-NE-SR", "0.791", "0.027", "M01-Q1-L4-NE-SR", "0.789",
          "0.027", "0.002", "0.041", "0.868", "0.679"),
    200: ("M2P-Q1-L2-NE-SR", "0.795", "0.027", "M2P-Q2-L4-E-SR", "0.793",
          "0.036", "0.001", "0.043", "0.925", "0.934"),
    500: ("M2P-Q2-L6-E-SR", "0.811", "0.025", "M2P-Q2-L4-E-AR", "0.808",
          "0.024", "0.003", "0.027", "0.689", "0.762"),
    1000: ("M2P-Q1-L4-NE-SR", "0.818", "0.022", "M2P-Q2-L6-E-AR", "0.808",
           "0.020", "0.009", "0.021", "0.099", "0.107"),
}
for tr, e in expect.items():
    r = t2[t2.train == tr].iloc[0]
    ok = (r.config_1 == e[0] and f"{r.config_1_mean:.3f}" == e[1]
          and f"{r.config_1_sd:.3f}" == e[2] and r.config_2 == e[3]
          and f"{r.config_2_mean:.3f}" == e[4] and f"{r.config_2_sd:.3f}" == e[5]
          and f"{r.diff_mean:.3f}" == e[6] and f"{r.diff_sd:.3f}" == e[7]
          and f"{r.ttest_p:.3f}" == e[8] and f"{r.wilcoxon_p:.3f}" == e[9])
    chk(f"Table2 T={tr}", ok)
    # also recompute mean/sd directly from fold rows
    sub = df[df.train == tr]
    m1 = sub[sub.config == e[0]].balanced_acc
    chk(f"Table2 T={tr} rank1 recompute",
        f"{m1.mean():.3f}" == e[1] and f"{m1.std(ddof=1):.3f}" == e[2],
        f"{m1.mean():.4f} +/- {m1.std(ddof=1):.4f}")

# T2 trend claim: rank1 means increase with T
means = [t2[t2.train == tr].config_1_mean.iloc[0] for tr in [100, 200, 500, 1000]]
chk("rank1 mean increases with T", all(np.diff(means) > 0),
    " -> ".join(f"{m:.3f}" for m in means))

# winner config recall/spec pooled (Table 3 QRGS row)
w = df[(df.train == 1000) & (df.config == "M2P-Q1-L4-NE-SR")]
chk("QRGS row Table3 0.818/0.835/0.800 SDs 0.022/0.043/0.045",
    f"{w.balanced_acc.mean():.3f}" == "0.818"
    and f"{w.balanced_acc.std(ddof=1):.3f}" == "0.022"
    and f"{w.recall.mean():.3f}" == "0.835"
    and f"{w.recall.std(ddof=1):.3f}" == "0.043"
    and f"{w.specificity.mean():.3f}" == "0.800"
    and f"{w.specificity.std(ddof=1):.3f}" == "0.045",
    f"BA {w.balanced_acc.mean():.4f}+/-{w.balanced_acc.std(ddof=1):.4f} "
    f"R {w.recall.mean():.4f}+/-{w.recall.std(ddof=1):.4f} "
    f"S {w.specificity.mean():.4f}+/-{w.specificity.std(ddof=1):.4f}")

print(f"{'CHECK':62s} RESULT")
print("-" * 80)
for name, ok, detail in checks:
    print(f"{name:62s} {'PASS' if ok else '*** FAIL ***'}  {detail}")
fails = [c for c in checks if not c[1]]
print("-" * 80)
print(f"{len(checks)} checks, {len(fails)} failures")
