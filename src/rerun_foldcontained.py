#!/usr/bin/env python3
"""
Fold-contained, budget-scaled QRGS runner for the manuscript revision.

Fixes vs. the original pipeline (advisor comments #1, #2, #4a):
  - PCA + class balancing are fit INSIDE each training fold only (no leakage).
    Test fold is balanced from test-only points and transformed with the
    train-fit PCA + scaler.
  - COBYLA budget scales with parameter count: epochs = K*(N+1), N = c*Q*L.
  - Continuous <Z> decision scores are saved per test sample -> ROC-AUC, PR-AUC,
    FPR@recall, cost curves. Per-iteration COBYLA loss saved -> convergence curves.

Reuses the EXACT circuit builder (create_quantum_reup_circuit), weight-count
formula, EstimatorQNN + NeuralNetworkClassifier + COBYLA training, and the
benign=-1 / malicious=+1 label convention from the published pipeline, so the
only methodological changes are the three above.

Fold construction (addresses comment #4c): for each seed, positive and negative
indices are shuffled with a seeded RNG and carved into n_splits MUTUALLY DISJOINT
chunks; within a chunk the first train_per_class go to train and the next
test_per_class to test. Folds are a function of (seed, sizes) ONLY, so all 72
configurations share identical fold assignments (paired tests remain valid).
"""
import argparse, json, math, os, time
import numpy as np

# ---- import the published method components (run from the src dir) -----------
from qiskit import transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
from sklearn.decomposition import PCA
from sklearn.metrics import (confusion_matrix, roc_auc_score,
                             average_precision_score)

from core.quantum_reup import create_quantum_reup_circuit
from preprocessing.methods import get_scaler
from preprocessing import doh

PCA_COMPONENTS = 6


# ----------------------------------------------------------------------------- raw data
def load_raw(dataset_path):
    """Return cleaned, imputed 28-feature matrix and labels in {-1,+1}
    (malicious=+1, benign=-1). NO global balancing, NO global PCA."""
    args = {"dataset": "doh", "dataset_path": dataset_path, "label_column": "Label",
            "positive_label": "Malicious", "balance_dataset": False,
            "feature_select": False, "feature_select_number": 0}
    df = doh.prepare_data(os.path.dirname(dataset_path), args)
    X = df.iloc[:, :-1].to_numpy(dtype=float)
    cls = df.iloc[:, -1].to_numpy().astype(int)          # 0=benign, 1=malicious
    y = np.where(cls == 0, -1, 1).astype(int)
    return X, y


def num_inputs_for(q, method):
    """Padded input dimension after PCA(6), matching the published padding."""
    n = PCA_COMPONENTS
    if method == "asymmetrical" and q > 1:
        block = 3 * q
        rem = n % block
        return n + (block - rem if rem else 0)
    block = 3                                             # symmetrical
    return int(math.ceil(n / block) * block)


def num_weights_for(num_inputs, q, layers, method):
    if q > 1 and method == "asymmetrical":
        return (num_inputs * 2) * layers
    return (num_inputs * 2) * layers * q


def make_folds(y, seed, n_splits, train_per_class, test_per_class):
    """Mutually-disjoint, class-balanced folds, deterministic in seed only."""
    rng = np.random.default_rng(int(seed))
    pos = np.where(y == 1)[0]; neg = np.where(y == -1)[0]
    rng.shuffle(pos); rng.shuffle(neg)
    per = train_per_class + test_per_class
    need = n_splits * per
    if len(pos) < need or len(neg) < need:
        raise ValueError(f"Not enough samples: need {need}/class, have "
                         f"{len(pos)} pos / {len(neg)} neg")
    folds = []
    for k in range(n_splits):
        pc = pos[k * per:(k + 1) * per]; nc = neg[k * per:(k + 1) * per]
        tr = np.concatenate([pc[:train_per_class], nc[:train_per_class]])
        te = np.concatenate([pc[train_per_class:], nc[train_per_class:]])
        folds.append((tr, te))
    return folds


def prep_fold(X, tr, te, q, method, scaler_name, seed):
    """PCA + pad + scale, all fit on the TRAINING fold only."""
    assert len(np.intersect1d(tr, te)) == 0, "train/test overlap!"
    pca = PCA(n_components=PCA_COMPONENTS, random_state=int(seed))
    Xtr = pca.fit_transform(X[tr]); Xte = pca.transform(X[te])
    ni = num_inputs_for(q, method); pad = ni - Xtr.shape[1]
    if pad > 0:
        Xtr = np.pad(Xtr, ((0, 0), (0, pad))); Xte = np.pad(Xte, ((0, 0), (0, pad)))
    scaler = get_scaler(scaler_name)
    Xtr = scaler.fit_transform(Xtr); Xte = scaler.transform(Xte)
    return Xtr, Xte, ni


def metrics_from(y_true_pm, y_score):
    """y_true_pm in {-1,+1}; y_score = <Z> (higher -> malicious=+1)."""
    y01 = (np.asarray(y_true_pm) == 1).astype(int)        # malicious = 1
    y_pred01 = (np.asarray(y_score) > 0).astype(int)      # sign(<Z>), sign(0)->benign
    tn, fp, fn, tp = confusion_matrix(y01, y_pred01, labels=[0, 1]).ravel()
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn)
    ba = 0.5 * (recall + spec)
    f1 = (2 * prec * recall / (prec + recall)) if (prec + recall) else 0.0
    out = dict(true_negative=int(tn), false_positive=int(fp),
               false_negative=int(fn), true_positive=int(tp),
               recall=recall, specificity=spec, precision=prec, acc=acc,
               balanced_acc=ba, f1=f1)
    # threshold-independent (need both classes present)
    if len(np.unique(y01)) == 2:
        out["roc_auc"] = float(roc_auc_score(y01, y_score))
        out["pr_auc"] = float(average_precision_score(y01, y_score))
    return out


def run_config(X, y, args):
    q, L = args.q_num_qubit, args.q_num_layers
    method, scaler = args.q_reup_method, args.feature_scaler
    N = num_weights_for(num_inputs_for(q, method), q, L, method)
    folds = make_folds(y, args.seed, args.n_splits,
                       args.train_count // 2, args.test_count // 2)

    ni = num_inputs_for(q, method)
    inputs = ParameterVector("input", ni)
    weights = ParameterVector("weight", num_weights_for(ni, q, L, method))
    ent_bool = str(args.q_entanglement).strip().lower() in ("true", "1", "yes")
    cdict = {"q_num_qubit": q, "q_num_layers": L, "q_entanglement": ent_bool,
             "q_reup_method": method}
    algorithm_globals.random_seed = int(args.seed)
    qc = create_quantum_reup_circuit(num_inputs=ni, inputs=inputs,
                                     weights=weights, args_dict=cdict)
    backend = AerSimulator(method="statevector")
    qc = transpile(qc, backend, optimization_level=3)
    qnn = EstimatorQNN(circuit=qc, input_params=inputs, weight_params=weights)

    per_fold, loss_curves, score_dump = [], [], []
    for k, (tr, te) in enumerate(folds):
        Xtr, Xte, _ = prep_fold(X, tr, te, q, method, scaler, args.seed)
        ytr, yte = y[tr], y[te]
        losses = []
        clf = NeuralNetworkClassifier(
            qnn, optimizer=COBYLA(maxiter=args.epochs),
            callback=lambda w, f: losses.append(float(f)))
        t0 = time.time()
        clf.fit(Xtr, ytr)
        ftime = time.time() - t0
        scores = np.asarray(qnn.forward(Xte, clf.weights)).reshape(-1)
        m = metrics_from(yte, scores)
        m["fold_train_seconds"] = round(ftime, 2)
        m["n_evals"] = len(losses)
        per_fold.append(m)
        loss_curves.append(losses)
        score_dump.append({"y_true": yte.tolist(), "score": scores.tolist()})

    keys = [k for k in per_fold[0] if isinstance(per_fold[0][k], float)]
    summary = {"nb_trainable_params": int(N), "q_num_qubit": q, "q_num_layers": L,
               "q_entanglement": args.q_entanglement, "q_reup_method": method,
               "feature_scaler": scaler, "epochs": args.epochs, "seed": args.seed,
               "train_count": args.train_count, "test_count": args.test_count}
    for k in keys:
        vals = [f[k] for f in per_fold]
        summary[f"cv_mean_{k}"] = float(np.mean(vals))
        summary[f"cv_SD_{k}"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return {"summary": summary, "per_fold": per_fold}, loss_curves, score_dump


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--raw_dir", required=True, help="dir for cached X.npy/y.npy")
    ap.add_argument("--prep", action="store_true", help="build raw cache and exit")
    ap.add_argument("--run_dir", help="output dir for this config")
    ap.add_argument("--feature_scaler"); ap.add_argument("--q_num_qubit", type=int)
    ap.add_argument("--q_num_layers", type=int); ap.add_argument("--q_entanglement")
    ap.add_argument("--q_reup_method"); ap.add_argument("--epochs", type=int)
    ap.add_argument("--seed", type=int); ap.add_argument("--train_count", type=int)
    ap.add_argument("--test_count", type=int, default=4000)
    ap.add_argument("--n_splits", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(args.raw_dir, exist_ok=True)
    xp = os.path.join(args.raw_dir, "X.npy"); yp = os.path.join(args.raw_dir, "y.npy")
    if args.prep:
        X, y = load_raw(args.dataset_path)
        np.save(xp, X); np.save(yp, y)
        print(f"raw cache: X{X.shape} y{y.shape} pos={(y==1).sum()} neg={(y==-1).sum()}")
        return

    X = np.load(xp); y = np.load(yp)
    os.makedirs(args.run_dir, exist_ok=True)
    res, loss_curves, score_dump = run_config(X, y, args)
    with open(os.path.join(args.run_dir, "all_metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(args.run_dir, "train_loss.json"), "w") as f:
        json.dump(loss_curves, f)
    with open(os.path.join(args.run_dir, "scores.json"), "w") as f:
        json.dump(score_dump, f)
    print("OK", res["summary"].get("cv_mean_balanced_acc"))


if __name__ == "__main__":
    main()
