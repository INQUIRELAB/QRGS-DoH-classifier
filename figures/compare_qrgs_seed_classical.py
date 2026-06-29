"""Compare per-seed QRGS winners with classical baselines.

For every training-size/seed pair, this script:
1. reads the QRGS grid-search summary and selects the best QRGS configuration;
2. rebuilds the same three disjoint folds used by QRGS;
3. evaluates classical baselines on the same six-component PCA data;
4. writes a detailed CSV and a compact best-vs-best CSV.

Run with the qnn-reup conda environment when TensorFlow/Keras MLP results are
needed:

    python analysis/compare_qrgs_seed_classical.py
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import random
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

try:
    import tensorflow as tf
    from tensorflow.keras import backend as keras_backend
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.models import Sequential
except Exception:  # pragma: no cover - local fallback when TensorFlow is absent
    tf = None
    keras_backend = None
    Dense = Input = Sequential = None


TRAINS = (100, 200, 500, 1000)
SEEDS = (7, 13, 37, 42, 101)
N_SPLITS = 3
TEST_COUNT = 4000
CLASSICAL_SCALER = "standard"
CLASSICAL_MODELS = ("LR", "LinSVC", "SmallMLP", "RBF_SVM")


def default_project_root() -> Path:
    # Override with --project-root; defaults to the current working directory.
    return Path(os.environ.get("QRGS_PROJECT_ROOT", "."))


def config_name(row: pd.Series) -> str:
    scaler = "M2P" if row["p_feature_scaler"] == "min0max2pi" else "M01"
    qubits = int(float(row["p_q_num_qubit"]))
    layers = int(float(row["p_q_num_layers"]))
    entanglement = "E" if str(row["p_q_entanglement"]).lower() == "true" else "NE"
    reupload = "SR" if row["p_q_reup_method"] == "symmetrical" else "AR"
    return f"{scaler}-Q{qubits}-L{layers}-{entanglement}-{reupload}"


def processed_dataset_path(results_root: Path, seed: int) -> Path:
    # The processed six-component PCA data depends on seed, not training size.
    # Use the same canonical QRGS run folder for each seed.
    pattern = (
        results_root
        / "100"
        / "src"
        / "results"
        / f"seed_{seed}"
        / "grid_search"
        / "gs_sc-min0max1_qb-1_ly-2_en-false_mt-symmetrical"
        / "processed_dataset.csv"
    )
    if pattern.exists():
        return pattern

    fallback = next(
        (
            p
            for p in (
                results_root
                / "100"
                / "src"
                / "results"
                / f"seed_{seed}"
                / "grid_search"
            ).glob("*/processed_dataset.csv")
        ),
        None,
    )
    if fallback is None:
        raise FileNotFoundError(f"No processed_dataset.csv found for seed {seed}")
    return fallback


def load_seed_data(results_root: Path, seed: int) -> tuple[np.ndarray, np.ndarray]:
    path = processed_dataset_path(results_root, seed)
    df = pd.read_csv(path, header=None)
    x = df.iloc[:, :-1].to_numpy(dtype=np.float64)
    y_raw = df.iloc[:, -1].to_numpy(dtype=int)
    y = (y_raw == 1).astype(int)
    return x, y


def split_indices(n_rows: int, train_count: int, seed: int):
    for fold in range(N_SPLITS):
        rng = np.random.RandomState(seed + fold)
        shuffled = rng.permutation(n_rows)
        train_idx = shuffled[:train_count]
        test_idx = shuffled[train_count : train_count + TEST_COUNT]
        yield fold + 1, train_idx, test_idx


def best_qrgs_row(results_root: Path, train_count: int, seed: int) -> dict[str, object]:
    path = (
        results_root
        / str(train_count)
        / "src"
        / "results"
        / f"seed_{seed}"
        / "grid_search"
        / "grid_search_summary.csv"
    )
    df = pd.read_csv(path)
    df = df[df["status"] == "success"].copy()
    if df.empty:
        raise ValueError(f"No successful QRGS rows in {path}")
    df["config"] = df.apply(config_name, axis=1)
    row = df.sort_values("m_cv_mean_balanced_acc", ascending=False).iloc[0]
    return {
        "qrgs_config": row["config"],
        "qrgs_balanced_acc_mean": float(row["m_cv_mean_balanced_acc"]),
        "qrgs_balanced_acc_sd": float(row["m_cv_SD_balanced_acc"]),
        "qrgs_acc_mean": float(row["m_cv_mean_acc"]),
        "qrgs_acc_sd": float(row["m_cv_SD_acc"]),
        "qrgs_f1_mean": float(row["m_cv_mean_f1"]),
        "qrgs_f1_sd": float(row["m_cv_SD_f1"]),
        "qrgs_precision_mean": float(row["m_cv_mean_precision"]),
        "qrgs_precision_sd": float(row["m_cv_SD_precision"]),
        "qrgs_recall_mean": float(row["m_cv_mean_recall"]),
        "qrgs_recall_sd": float(row["m_cv_SD_recall"]),
        "qrgs_specificity_mean": float(row["m_cv_mean_specificity"]),
        "qrgs_specificity_sd": float(row["m_cv_SD_specificity"]),
        "qrgs_params": int(float(row["m_nb_trainable_params"])),
    }


def classical_model(name: str, seed: int):
    if name == "LR":
        return LogisticRegression(max_iter=2000, random_state=seed)
    if name == "LinSVC":
        return LinearSVC(random_state=seed, max_iter=20000)
    if name == "RBF_SVM":
        return SVC(kernel="rbf", random_state=seed)
    raise ValueError(name)


def build_small_mlp(input_dim: int, seed: int):
    if tf is None or Sequential is None:
        raise RuntimeError("TensorFlow is required for SmallMLP; run in qnn-reup environment.")
    keras_backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(12, activation="relu"))
    model.add(Dense(2, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def parameter_count(model_name: str) -> int | str:
    if model_name == "SmallMLP":
        return 113
    return ""


def summarize_fold_metrics(metrics: list[dict[str, float]]) -> dict[str, object]:
    out: dict[str, object] = {}
    for metric in ("balanced_acc", "acc", "f1", "precision", "recall", "specificity", "far"):
        values = np.array([row[metric] for row in metrics], dtype=float)
        out[f"classical_{metric}_mean"] = float(values.mean())
        out[f"classical_{metric}_sd"] = float(values.std(ddof=1))
    for key in ("tn", "fp", "fn", "tp"):
        out[f"classical_{key}"] = int(sum(row[key] for row in metrics))
    return out


def evaluate_classical(
    x: np.ndarray,
    y: np.ndarray,
    train_count: int,
    seed: int,
    model_name: str,
    epochs: int,
) -> dict[str, object]:
    fold_metrics: list[dict[str, float]] = []
    for fold, train_idx, test_idx in split_indices(len(x), train_count, seed):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        if model_name == "SmallMLP":
            model = build_small_mlp(x_train.shape[1], seed + fold)
            model.fit(
                x_train,
                y_train,
                epochs=epochs,
                batch_size=32,
                verbose=0,
                validation_split=0.2,
            )
            y_pred = (model.predict(x_test, verbose=0).reshape(-1) > 0.5).astype(int)
        else:
            model = classical_model(model_name, seed)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        fold_metrics.append(
            {
                "fold": fold,
                "balanced_acc": balanced_accuracy_score(y_test, y_pred),
                "acc": accuracy_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "specificity": tn / (tn + fp),
                "far": fp / (tn + fp),
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        )

    summary = summarize_fold_metrics(fold_metrics)
    summary["classical_model"] = model_name
    summary["classical_params"] = parameter_count(model_name)
    summary["classical_scaler"] = CLASSICAL_SCALER
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=default_project_root())
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    if tf is not None:
        tf.get_logger().setLevel("ERROR")

    results_root = args.project_root / "DoH_Seed"
    out_dir = args.out_dir or results_root / "analysis" / "classical_comparison"

    detailed_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    data_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for seed in SEEDS:
        print(f"Loading processed data for seed {seed}")
        data_cache[seed] = load_seed_data(results_root, seed)

        for train_count in TRAINS:
            print(f"Evaluating train={train_count}, seed={seed}")
            qrgs = best_qrgs_row(results_root, train_count, seed)
            x, y = data_cache[seed]
            pair_rows: list[dict[str, object]] = []

            for model_name in CLASSICAL_MODELS:
                classical = evaluate_classical(x, y, train_count, seed, model_name, args.epochs)
                row = {
                    "train_count": train_count,
                    "seed": seed,
                    "n_splits": N_SPLITS,
                    "test_count": TEST_COUNT,
                    **qrgs,
                    **classical,
                }
                row["delta_qrgs_minus_classical_balanced_acc"] = (
                    row["qrgs_balanced_acc_mean"] - row["classical_balanced_acc_mean"]
                )
                pair_rows.append(row)

            best = max(pair_rows, key=lambda r: float(r["classical_balanced_acc_mean"]))
            for row in pair_rows:
                row["is_best_classical_for_seed_train"] = row["classical_model"] == best["classical_model"]
                detailed_rows.append(row)

            summary_rows.append(
                {
                    "train_count": train_count,
                    "seed": seed,
                    "qrgs_best_config": qrgs["qrgs_config"],
                    "qrgs_balanced_acc_mean": qrgs["qrgs_balanced_acc_mean"],
                    "qrgs_balanced_acc_sd": qrgs["qrgs_balanced_acc_sd"],
                    "qrgs_recall_mean": qrgs["qrgs_recall_mean"],
                    "qrgs_specificity_mean": qrgs["qrgs_specificity_mean"],
                    "qrgs_params": qrgs["qrgs_params"],
                    "best_classical_model": best["classical_model"],
                    "best_classical_balanced_acc_mean": best["classical_balanced_acc_mean"],
                    "best_classical_balanced_acc_sd": best["classical_balanced_acc_sd"],
                    "best_classical_recall_mean": best["classical_recall_mean"],
                    "best_classical_specificity_mean": best["classical_specificity_mean"],
                    "best_classical_params": best["classical_params"],
                    "classical_scaler": CLASSICAL_SCALER,
                    "delta_qrgs_minus_best_classical_balanced_acc": best[
                        "delta_qrgs_minus_classical_balanced_acc"
                    ],
                }
            )

    detailed_rows.sort(key=lambda r: (int(r["train_count"]), int(r["seed"]), str(r["classical_model"])))
    summary_rows.sort(key=lambda r: (int(r["train_count"]), int(r["seed"])))

    detailed_path = out_dir / "qrgs_vs_classical_all_algorithms_by_seed_train.csv"
    summary_path = out_dir / "best_qrgs_vs_best_classical_by_seed_train.csv"
    write_csv(detailed_path, detailed_rows)
    write_csv(summary_path, summary_rows)
    print(f"Wrote {detailed_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
