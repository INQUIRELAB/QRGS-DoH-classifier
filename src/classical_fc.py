#!/usr/bin/env python3
"""
Fold-contained classical + Fourier-feature baselines on the SAME folds/PCA as the
QRGS re-run (revision comments #3, #4a). For each (size, seed, fold): identical
balanced disjoint folds (seeded), PCA(6) fit on the training fold, StandardScaler
on train; then fit each classical model and a fixed-frequency Fourier-feature
logistic model. Reports BA / recall / specificity / ROC-AUC / PR-AUC aggregated
over 5 seeds x 3 folds, per training size.
"""
import csv, os, numpy as np
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

RAW = os.path.expanduser("~/qrgs_rerun/raw")
SEEDS = [7, 13, 37, 42, 101]
SIZES = [100, 200, 500, 1000]
PCAK = 6


def make_folds(y, seed, n_splits, train_per_class, test_per_class):
    rng = np.random.default_rng(int(seed))
    pos = np.where(y == 1)[0]; neg = np.where(y == -1)[0]
    rng.shuffle(pos); rng.shuffle(neg)
    per = train_per_class + test_per_class
    folds = []
    for k in range(n_splits):
        pc = pos[k * per:(k + 1) * per]; nc = neg[k * per:(k + 1) * per]
        tr = np.concatenate([pc[:train_per_class], nc[:train_per_class]])
        te = np.concatenate([pc[train_per_class:], nc[train_per_class:]])
        folds.append((tr, te))
    return folds


def fourier_features(X, K):
    """Fixed-frequency trig feature map: [x, sin(kx), cos(kx)] for k=1..K, per dim."""
    feats = [X]
    for k in range(1, K + 1):
        feats.append(np.sin(k * X)); feats.append(np.cos(k * X))
    return np.hstack(feats)


def metrics(y_true_pm, score):
    y01 = (y_true_pm == 1).astype(int)
    pred = (score > 0).astype(int) if score.min() < 0 else (score > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y01, pred, labels=[0, 1]).ravel()
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    ba = 0.5 * (rec + spec)
    out = dict(balanced_acc=ba, recall=rec, specificity=spec)
    if len(np.unique(y01)) == 2:
        out["roc_auc"] = roc_auc_score(y01, score)
        out["pr_auc"] = average_precision_score(y01, score)
    return out


def score_of(model, Xte):
    if hasattr(model, "decision_function"):
        return model.decision_function(Xte)
    return model.predict_proba(Xte)[:, 1]


def models():
    return {
        "LogReg": lambda: LogisticRegression(max_iter=2000),
        "LinearSVC": lambda: LinearSVC(C=1.0, max_iter=5000),
        "MLP": lambda: MLPClassifier(hidden_layer_sizes=(14,), max_iter=2000,
                                     random_state=0),   # ~113 params, matches paper
        "RBF_SVM": lambda: SVC(kernel="rbf", C=1.0, gamma="scale"),
        "Fourier_K2": "fourier2",
        "Fourier_K4": "fourier4",
    }


def main():
    X = np.load(os.path.join(RAW, "X.npy")); y = np.load(os.path.join(RAW, "y.npy"))
    rows = []
    for T in SIZES:
        agg = defaultdict(lambda: defaultdict(list))
        for seed in SEEDS:
            for tr, te in make_folds(y, seed, 3, T // 2, 2000):
                Xtr, Xte = X[tr], X[te]; ytr, yte = y[tr], y[te]
                pca = PCA(n_components=PCAK, random_state=int(seed))
                Ztr = pca.fit_transform(Xtr); Zte = pca.transform(Xte)
                sc = StandardScaler().fit(Ztr); Ztr_s = sc.transform(Ztr); Zte_s = sc.transform(Zte)
                for name, spec in models().items():
                    if spec == "fourier2" or spec == "fourier4":
                        K = 2 if spec == "fourier2" else 4
                        Ftr = fourier_features(Ztr_s, K); Fte = fourier_features(Zte_s, K)
                        m = LogisticRegression(max_iter=3000).fit(Ftr, (ytr == 1).astype(int))
                        s = m.decision_function(Fte)
                    else:
                        m = spec().fit(Ztr_s, (ytr == 1).astype(int))
                        s = score_of(m, Zte_s)
                    for k, v in metrics(yte, s).items():
                        agg[name][k].append(v)
        for name, m in agg.items():
            r = dict(train=T, model=name, n_obs=len(m["balanced_acc"]))
            for k, sh in [("balanced_acc", "ba"), ("recall", "rec"), ("specificity", "spec"),
                          ("roc_auc", "auc"), ("pr_auc", "prauc")]:
                v = m[k]; r[sh + "_mean"] = round(float(np.mean(v)), 4); r[sh + "_sd"] = round(float(np.std(v, ddof=1)), 4)
            rows.append(r)
    out = os.environ.get("QRGS_CLASSICAL_OUT", "classical_fc_aggregated.csv")
    with open(out, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", out, "rows", len(rows))
    for T in SIZES:
        print(f"--- T={T} ---")
        for r in sorted([x for x in rows if x["train"] == T], key=lambda x: -x["ba_mean"]):
            print(f"  {r['model']:11s} BA {r['ba_mean']:.3f}±{r['ba_sd']:.3f}  AUC {r.get('auc_mean','-')}  PRAUC {r.get('prauc_mean','-')}")


if __name__ == "__main__":
    main()
