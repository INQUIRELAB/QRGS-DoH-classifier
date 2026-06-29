import logging
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def _resolve_csv_path(data_dir: str, args_dict: dict) -> str:
    """
    Resolve CSV path from CLI args, falling back to src/data/BCCC-CIRA-CIC-DoHBrw-2020.csv.

    Supported args:
      - dataset_path: absolute or relative path to a CSV file
    """
    dataset_path = args_dict.get("dataset_path") or args_dict.get("doh_path")
    if dataset_path:
        if os.path.isabs(dataset_path):
            return dataset_path

        # Try relative to current working directory first (common when running from repo root)
        cwd_candidate = os.path.abspath(dataset_path)
        if os.path.exists(cwd_candidate):
            return cwd_candidate

        # Then try relative to src/ (args_dict['parent_directory']) if present
        parent_directory = args_dict.get("parent_directory")
        if parent_directory:
            parent_candidate = os.path.abspath(os.path.join(parent_directory, dataset_path))
            if os.path.exists(parent_candidate):
                return parent_candidate

        # Finally, try relative to src/data/
        data_candidate = os.path.abspath(os.path.join(data_dir, dataset_path))
        return data_candidate

    return os.path.join(data_dir, "BCCC-CIRA-CIC-DoHBrw-2020.csv")


def _pick_label_column(df: pd.DataFrame, preferred: Optional[str] = None) -> str:
    if preferred:
        if preferred in df.columns:
            return preferred
        # case-insensitive match
        lower_map = {str(c).lower(): c for c in df.columns}
        if preferred.lower() in lower_map:
            return lower_map[preferred.lower()]
        raise ValueError(f"label_column '{preferred}' not found in CSV columns: {list(df.columns)}")

    candidates = [
        "Class",
        "class",
        "label",
        "Label",
        "target",
        "Target",
        "y",
        "Y",
        "is_doh",
        "isDoH",
        "doh",
        "DoH",
    ]
    lower_map = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    # Fall back to last column
    return df.columns[-1]


def _coerce_binary_labels(y_raw: pd.Series, positive_label: Optional[str] = None) -> np.ndarray:
    """
    Convert a label series into {0,1} ints.
    Handles numeric labels, booleans, and common string labels.
    """
    # Fast path: numeric already
    if pd.api.types.is_numeric_dtype(y_raw):
        y = pd.to_numeric(y_raw, errors="coerce")
        uniq = set(pd.Series(y).dropna().unique().tolist())
        if uniq.issubset({0, 1}):
            return y.fillna(0).astype(int).to_numpy()
        if uniq.issubset({-1, 1}):
            return y.map(lambda v: 0 if v == -1 else 1).fillna(0).astype(int).to_numpy()
        # If numeric but not binary, threshold at median (best-effort)
        med = float(np.nanmedian(y.to_numpy(dtype=float)))
        return (y.fillna(med) > med).astype(int).to_numpy()

    # Otherwise treat as strings / booleans
    s = y_raw.astype(str).str.strip()
    s_lower = s.str.lower()

    if positive_label:
        pos = positive_label.strip().lower()
        return (s_lower == pos).astype(int).to_numpy()

    # Common binary tokens
    pos_tokens = {"1", "true", "t", "yes", "y", "doh", "dns-over-https", "dns over https", "encrypted", "malicious"}
    neg_tokens = {"0", "false", "f", "no", "n", "non-doh", "nodoh", "plain", "unencrypted", "benign"}

    # If tokens match, map them
    if set(s_lower.unique()).issubset(pos_tokens | neg_tokens):
        return s_lower.map(lambda v: 1 if v in pos_tokens else 0).astype(int).to_numpy()

    # Best-effort: malicious/attack traffic is positive
    return s_lower.str.contains(r"\b(malicious|attack|doh)\b", regex=True).astype(int).to_numpy()


def _coerce_feature_frame(df_x: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Keep/convert features to numeric.
    - Numeric columns kept
    - Non-numeric columns: try to coerce to numeric; otherwise drop
    Returns (numeric_df, dropped_column_count)
    """
    numeric_df = df_x.select_dtypes(include=[np.number]).copy()
    dropped = 0

    for col in df_x.columns:
        if col in numeric_df.columns:
            continue
        coerced = pd.to_numeric(df_x[col], errors="coerce")
        non_nan_ratio = float(coerced.notna().mean())
        if non_nan_ratio >= 0.95:
            numeric_df[col] = coerced
        else:
            dropped += 1

    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return numeric_df, dropped


def prepare_data(data_dir: str, args_dict: dict) -> pd.DataFrame:
    """
    Prepare the BCCC-CIRA-CIC-DoHBrw-2020 CSV dataset for this repo:
    - Reads the CSV (from --dataset_path or src/data/BCCC-CIRA-CIC-DoHBrw-2020.csv)
    - Extracts labels -> column 'Class' (0=Benign, 1=Malicious)
    - Converts features to numeric, drops non-numeric columns that can't be coerced
    - Renames feature columns to input_1..input_n and keeps label last
    """
    csv_path = _resolve_csv_path(data_dir, args_dict)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV not found at '{csv_path}'. "
            f"Upload/copy your dataset into '{data_dir}' (e.g. '{os.path.join(data_dir, 'BCCC-CIRA-CIC-DoHBrw-2020.csv')}') "
            f"or pass --dataset_path /path/to/your.csv"
        )

    logging.info(f"Loading dataset CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns (features + label). Got shape={df.shape}")

    label_col = _pick_label_column(df, preferred=args_dict.get("label_column"))
    y = _coerce_binary_labels(df[label_col], positive_label=args_dict.get("positive_label"))

    df_x = df.drop(columns=[label_col])
    df_x, dropped = _coerce_feature_frame(df_x)

    if df_x.shape[1] == 0:
        raise ValueError(
            "After numeric coercion, no usable feature columns remained. "
            "If your CSV has categorical features you want encoded, we can add one-hot encoding."
        )

    if dropped:
        logging.info(f"Dropped {dropped} non-numeric feature columns (could not safely coerce).")

    # Optional: select specific named features before renaming
    feature_select_names = args_dict.get("feature_select_names")
    if feature_select_names:
        # Support both list and comma-separated string
        if isinstance(feature_select_names, str):
            feature_select_names = [f.strip() for f in feature_select_names.split(",")]
        missing = [f for f in feature_select_names if f not in df_x.columns]
        if missing:
            raise ValueError(
                f"feature_select_names: the following columns were not found in the CSV: {missing}. "
                f"Available columns: {list(df_x.columns)}"
            )
        df_x = df_x[feature_select_names].copy()
        logging.info(f"* Selected {len(feature_select_names)} named features: {feature_select_names}")

    x_np = df_x.to_numpy(dtype=float)
    out = pd.DataFrame(x_np, columns=[f"input_{i+1}" for i in range(x_np.shape[1])])
    out["Class"] = y.astype(int)

    # Balance the dataset: keep all positive (Malicious) rows and randomly
    # sample an equal number of negative (Benign) rows.
    # Controlled by args_dict.get("balance_dataset", True).
    # Optionally cap each class at max_samples_per_class rows.
    if args_dict.get("balance_dataset", True):
        rng = args_dict.get("random_seed", 42)
        max_per_class = args_dict.get("max_samples_per_class", None)

        pos_df = out[out["Class"] == 1]
        neg_df = out[out["Class"] == 0]

        # Determine target count per class
        n_target = min(len(pos_df), len(neg_df))
        if max_per_class is not None:
            n_target = min(n_target, int(max_per_class))

        pos_df = pos_df.sample(n=n_target, random_state=rng)
        neg_df = neg_df.sample(n=n_target, random_state=rng)

        logging.info(
            f"* Balanced dataset: {n_target} positive + {n_target} negative = {n_target * 2} total "
            f"(from {len(out)} original rows"
            + (f", capped at {max_per_class} per class)" if max_per_class else ")")
        )
        out = (
            pd.concat([pos_df, neg_df])
            .sample(frac=1, random_state=rng)
            .reset_index(drop=True)
        )
    else:
        logging.info(
            f"* Balancing skipped (balance_dataset=False): "
            f"{len(out[out['Class']==1])} positive + {len(out[out['Class']==0])} negative = {len(out)} total"
        )

    return out

