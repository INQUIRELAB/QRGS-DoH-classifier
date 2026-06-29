"""Aggregate QRGS seed-grid results for the manuscript.

This script pools the five requested random seeds and the three folds within
each seed. For each training size and QRGS configuration, it reports means and
sample standard deviations over the 15 seed-by-fold observations.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

try:
    from scipy import stats
except Exception:  # pragma: no cover - only used when SciPy is unavailable
    stats = None


TRAINS = (100, 200, 500, 1000)
SEEDS = (7, 13, 37, 42, 101)
FOLDS = (1, 2, 3)


def config_name(row: dict[str, str]) -> str:
    scaler = "M2P" if row["p_feature_scaler"] == "min0max2pi" else "M01"
    qubits = int(float(row["p_q_num_qubit"]))
    layers = int(float(row["p_q_num_layers"]))
    entanglement = "E" if str(row["p_q_entanglement"]).lower() == "true" else "NE"
    reupload = "SR" if row["p_q_reup_method"] == "symmetrical" else "AR"
    return f"{scaler}-Q{qubits}-L{layers}-{entanglement}-{reupload}"


def read_fold_records(results_root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for train in TRAINS:
        for seed in SEEDS:
            summary_path = (
                results_root
                / str(train)
                / "src"
                / "results"
                / f"seed_{seed}"
                / "grid_search"
                / "grid_search_summary.csv"
            )
            with summary_path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    if row["status"] != "success":
                        continue
                    cfg = config_name(row)
                    for fold in FOLDS:
                        tn = float(row[f"m_f{fold}_true_negative"])
                        fp = float(row[f"m_f{fold}_false_positive"])
                        fn = float(row[f"m_f{fold}_false_negative"])
                        tp = float(row[f"m_f{fold}_true_positive"])
                        records.append(
                            {
                                "train": train,
                                "seed": seed,
                                "fold": fold,
                                "config": cfg,
                                "params": int(float(row["m_nb_trainable_params"])),
                                "balanced_acc": float(row[f"m_f{fold}_balanced_acc"]),
                                "acc": float(row[f"m_f{fold}_acc"]),
                                "f1": float(row[f"m_f{fold}_f1"]),
                                "precision": float(row[f"m_f{fold}_precision"]),
                                "recall": float(row[f"m_f{fold}_recall"]),
                                "specificity": tn / (tn + fp),
                                "far": fp / (tn + fp),
                                "tn": tn,
                                "fp": fp,
                                "fn": fn,
                                "tp": tp,
                            }
                        )
    return records


def summarize(records: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[(int(record["train"]), str(record["config"]))].append(record)

    summary_rows: list[dict[str, object]] = []
    for (train, cfg), values in grouped.items():
        row: dict[str, object] = {
            "train": train,
            "config": cfg,
            "n_seed_fold": len(values),
            "params": values[0]["params"],
        }
        for metric in ("balanced_acc", "acc", "f1", "precision", "recall", "specificity", "far"):
            vals = [float(v[metric]) for v in values]
            row[f"{metric}_mean"] = mean(vals)
            row[f"{metric}_sd"] = stdev(vals)
        for metric in ("tn", "fp", "fn", "tp"):
            row[metric] = int(sum(float(v[metric]) for v in values))
        summary_rows.append(row)

    summary_rows.sort(key=lambda r: (int(r["train"]), -float(r["balanced_acc_mean"]), str(r["config"])))

    paired_rows: list[dict[str, object]] = []
    by_train: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        by_train[int(row["train"])].append(row)

    for train, rows in sorted(by_train.items()):
        first, second = rows[0], rows[1]
        first_values = sorted(
            grouped[(train, str(first["config"]))],
            key=lambda r: (int(r["seed"]), int(r["fold"])),
        )
        second_values = sorted(
            grouped[(train, str(second["config"]))],
            key=lambda r: (int(r["seed"]), int(r["fold"])),
        )
        a = [float(v["balanced_acc"]) for v in first_values]
        b = [float(v["balanced_acc"]) for v in second_values]
        diffs = [x - y for x, y in zip(a, b)]
        paired: dict[str, object] = {
            "train": train,
            "config_1": first["config"],
            "config_1_mean": first["balanced_acc_mean"],
            "config_1_sd": first["balanced_acc_sd"],
            "config_2": second["config"],
            "config_2_mean": second["balanced_acc_mean"],
            "config_2_sd": second["balanced_acc_sd"],
            "diff_mean": mean(diffs),
            "diff_sd": stdev(diffs),
            "n_pairs": len(diffs),
        }
        if stats is not None:
            paired["ttest_p"] = float(stats.ttest_rel(a, b).pvalue)
            paired["wilcoxon_p"] = float(
                stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided", method="auto").pvalue
            )
        paired_rows.append(paired)

    return summary_rows, paired_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results/results_fc"))
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "seed_aggregate")
    args = parser.parse_args()

    records = read_fold_records(args.results_root)
    summary_rows, paired_rows = summarize(records)

    write_csv(args.out_dir / "qrgs_seed_fold_summary.csv", summary_rows)
    write_csv(args.out_dir / "qrgs_top2_paired_tests.csv", paired_rows)
    with (args.out_dir / "qrgs_seed_fold_summary.json").open("w") as handle:
        json.dump({"summary": summary_rows, "paired_tests": paired_rows}, handle, indent=2)

    print(f"Read {len(records)} seed-fold records")
    print(f"Wrote {args.out_dir / 'qrgs_seed_fold_summary.csv'}")
    print(f"Wrote {args.out_dir / 'qrgs_top2_paired_tests.csv'}")
    for row in paired_rows:
        print(
            f"T={row['train']}: {row['config_1']} "
            f"{float(row['config_1_mean']):.3f} +/- {float(row['config_1_sd']):.3f} vs "
            f"{row['config_2']} {float(row['config_2_mean']):.3f} +/- "
            f"{float(row['config_2_sd']):.3f}; "
            f"t p={float(row.get('ttest_p', float('nan'))):.4f}, "
            f"Wilcoxon p={float(row.get('wilcoxon_p', float('nan'))):.4f}"
        )


if __name__ == "__main__":
    main()
