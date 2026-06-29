import argparse
import glob
import json
import logging
import os
import shutil
import time
from datetime import datetime
from uuid import uuid4

import mlflow
import numpy as np
import pandas as pd
import pytz

from core.functions import load_dataset
from helpers.helper import int_or_none, str2bool
from helpers.metrics import plot_confusion_matrix_from_counts


logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
handler = logging.StreamHandler()
logger = logging.getLogger(f"azlog_{__name__}")
logger.addHandler(handler)
logger.setLevel(logging.INFO)


if __name__ == "__main__":
    t_script_start = time.perf_counter()

    parser = argparse.ArgumentParser()

    # Generic CSV dataset options (used by --dataset doh)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--label_column", type=str, default=None)
    parser.add_argument("--positive_label", type=str, default=None)

    # Persistence options
    parser.add_argument("--keep_results", type=str2bool, default=True)
    parser.add_argument("--copy_processed_data_to_results", type=str2bool, default=True)
    parser.add_argument("--copy_raw_dataset_to_results", type=str2bool, default=True)

    # Naming / bookkeeping
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument(
        "--run_name_params",
        type=str,
        default=None,
        help="Comma-separated param names to build a run folder name (e.g., feature_select_number,feature_scaler).",
    )
    parser.add_argument("--results_root", type=str, default=None)

    parser.add_argument("--algorithm",type=str)

    parser.add_argument("--dataset",type=str)
    parser.add_argument("--downsample",type=str2bool)
    parser.add_argument("--feature_select",type=str2bool)
    parser.add_argument("--feature_select_number",type=int)
    parser.add_argument("--pca_components",type=int_or_none)
    parser.add_argument("--scale_features",type=str2bool)
    parser.add_argument("--feature_scaler",type=str)
    
    parser.add_argument("--test_split",type=float)
    parser.add_argument("--n_splits",type=int)
    parser.add_argument("--cv_method",type=str)
    # Exact-count split: when both are set, RandomDisjointSplit is used instead of cv_method
    parser.add_argument("--train_count", type=int_or_none, default=None,
                        help="Exact number of training samples per CV iteration.")
    parser.add_argument("--test_count",  type=int_or_none, default=None,
                        help="Exact number of test samples per CV iteration (no overlap with train).")
    
    parser.add_argument("--random_seed",type=int)
    parser.add_argument("--classifier_seed",type=int)
    parser.add_argument("--epochs",type=int)

    ## QUANTUM ARGUMENTS
    parser.add_argument("--q_random_state",type=int)
    parser.add_argument("--q_noise",type=str2bool)
    parser.add_argument("--q_input_gradients",type=str2bool)
    parser.add_argument("--q_feature_map_reps",type=int)
    parser.add_argument("--q_ansatz",type=str)
    parser.add_argument("--q_ansatz_reps",type=int)
    parser.add_argument("--q_optimization_level",type=int)
    parser.add_argument("--q_optimizer",type=str)
    parser.add_argument("--q_backend",type=str)
    parser.add_argument("--q_shots",type=int)
    
    ### QNN REUPLOAD ARGS
    parser.add_argument("--q_reup_method",type=str)
    parser.add_argument("--q_num_qubit",type=int)
    parser.add_argument("--q_num_layers",type=int)
    parser.add_argument("--q_entanglement",type=str2bool)

    args = parser.parse_args()

    timings = {}

    # MLFLOW START RUN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    t_mlflow_start = time.perf_counter()
    mlflow.start_run()
    timings["mlflow_start_seconds"] = time.perf_counter() - t_mlflow_start
    run = mlflow.active_run()

    # Get the current time in UTC
    utc_time = datetime.now()

    # Convert to Eastern Time
    eastern = pytz.timezone('US/Eastern')
    eastern_time = utc_time.astimezone(eastern)

    # Format the time string as YYYYMMDD_HHMMSS
    time_string = eastern_time.strftime('%Y%m%d_%H%M%S_%f')

    # run_id = run.info.run_id
    run_id = time_string

    # Parse CLI args into dict
    args_dict = vars(args)
    args_dict['parent_directory'] = os.path.dirname(os.path.abspath(__file__))
    # Inject class names derived from positive_label so classifiers can label CM axes
    pos = args_dict.get("positive_label") or "Malicious"
    neg = "Benign" if pos.lower() == "malicious" else f"non-{pos}"
    args_dict["class_names"] = [neg, pos]
    defaults_dict = {a.dest: a.default for a in parser._actions if hasattr(a, "dest")}
    # Normalize dataset key for internal routing (e.g., DoH -> doh)
    if args_dict.get("dataset") is not None:
        args_dict["dataset"] = str(args_dict["dataset"]).lower()
    run_label = f"{args_dict['algorithm']}_{args_dict['dataset']}"

    # Optional user-provided run name (used in results folder + MLflow run name)
    run_name = args_dict.get("run_name")
    if run_name:
        safe = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in str(run_name)).strip("_")
        if safe:
            # Use the parameter-driven run name as the folder name (no timestamp prefix)
            run_id = safe
            try:
                mlflow.set_tag("run_name", safe)
            except Exception:
                pass
        else:
            run_id = f"{time_string}_{uuid4().hex[:8]}"
    else:
        # Build run name from selected parameters when requested.
        run_name_params = args_dict.get("run_name_params")
        if run_name_params:
            parts = []
            for key in [p.strip() for p in str(run_name_params).split(",") if p.strip()]:
                if key in args_dict and args_dict[key] is not None:
                    val = args_dict[key]
                    safe_val = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in str(val)).strip("_")
                    parts.append(f"{key}_{safe_val}")
            if parts:
                run_id = "_".join(parts)
                try:
                    mlflow.set_tag("run_name", run_id)
                except Exception:
                    pass
        else:
            # Auto-name runs based on parameters that differ from defaults.
            name_parts = []
            # Keep auto-names short: only include key tunables.
            name_keys = [
                "feature_select_number",
                "pca_components",
                "feature_scaler",
                "test_split",
                "cv_method",
                "random_seed",
                "q_random_state",
                "q_optimization_level",
                "q_optimizer",
                "q_reup_method",
                "q_num_qubit",
                "q_num_layers",
                "q_entanglement",
            ]
            for key in name_keys:
                if key not in args_dict:
                    continue
                val = args_dict.get(key)
                default_val = defaults_dict.get(key)
                if key == "dataset_path" and val:
                    val = os.path.basename(str(val))
                if val is None:
                    continue
                if val != default_val:
                    safe_val = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in str(val)).strip("_")
                    name_parts.append(f"{key}_{safe_val}")
            if name_parts:
                run_id = "_".join(name_parts)
                # Avoid path-too-long errors by truncating with a stable hash suffix.
                if len(run_id) > 140:
                    short = run_id[:120].rstrip("_")
                    suffix = uuid4().hex[:8]
                    run_id = f"{short}_{suffix}"
                try:
                    mlflow.set_tag("run_name", run_id)
                except Exception:
                    pass

    logger.info(f"Reup Method: {args_dict['q_reup_method']}")

    # Set numpy and qiskit random seeds
    np.random.seed(args_dict['random_seed'])

    # Entanglement is hard-coded for 2, 4, and 5 qubits
    if args_dict['q_num_qubit'] not in [1,2,4,5]:
        raise Exception(f"For data reuploading, number of qubits must be 1, 2, 4, or 5. Currently set to {args_dict['q_num_qubit']}")

    # Log args as params
    mlflow.log_params(args_dict)

    results_dir = args_dict.get("results_root") or os.path.join(args_dict['parent_directory'], "results")
    os.makedirs(results_dir, exist_ok=True)

    data_dir = os.path.join(args_dict['parent_directory'], "data")
    os.makedirs(data_dir, exist_ok=True)

    run_results_dir = os.path.join(results_dir, run_id)
    if os.path.exists(run_results_dir):
        # Avoid collisions if the same run name is reused.
        run_id = f"{run_id}_{time_string}"
        run_results_dir = os.path.join(results_dir, run_id)
    os.makedirs(run_results_dir, exist_ok=True)

    # Persist run configuration immediately
    run_params_path = os.path.join(run_results_dir, "run_params.json")
    with open(run_params_path, "w") as f:
        json.dump(args_dict, f, indent=2, default=str)

    # Load Data
    filename = None
    processed_file_path = None
    metadata_file_path = None

    t_data_prep_start = time.perf_counter()
    filename, processed_file_path, metadata_file_path = load_dataset(
        data_dir=data_dir,
        args_dict=args_dict,
        run_results_dir=run_results_dir
    )
    timings["data_prepare_seconds"] = time.perf_counter() - t_data_prep_start

    # Log data params
    with open(metadata_file_path) as dataset_metadata_file:
        dataset_metadata = json.load(dataset_metadata_file)
        mlflow.log_params(dataset_metadata)

    # Snapshot dataset metadata into the run folder (so it's not overwritten across runs)
    dataset_metadata_snapshot_path = os.path.join(run_results_dir, "dataset_metadata.json")
    with open(dataset_metadata_snapshot_path, "w") as f:
        json.dump(dataset_metadata, f, indent=2)

    # Read in processed data
    t_processed_read_start = time.perf_counter()
    df = pd.read_csv(processed_file_path, header=None)
    timings["processed_csv_read_seconds"] = time.perf_counter() - t_processed_read_start

    # save data as X and labels as Y into numpy arrays
    X = df.iloc[:, :-1].to_numpy()
    Y = df.iloc[:, -1:].to_numpy().astype("int").flatten()

    # Run selected classifier (lazy imports so classical runs do not require Qiskit)
    result, cv_results = None, None
    t_train_start = time.perf_counter()
    if args_dict['algorithm'] == "xgb":
        from algorithms import xgb_classifier

        result, cv_results = xgb_classifier.run_xgb_classifier_binary(
            run_label=run_label,
            run_id=run_id,
            run_results_dir=run_results_dir,
            X=X,
            y=Y,
            args_dict=args_dict
        )
    elif args_dict['algorithm'] == "nn":
        from algorithms import nn_classifier

        result, cv_results = nn_classifier.run_nn_classifier_binary(
            run_label=run_label,
            run_id=run_id,
            run_results_dir=run_results_dir,
            X=X,
            y=Y,
            args_dict=args_dict
        )
    elif args_dict['algorithm'] == "lr":
        from algorithms import lr_classifier

        result, cv_results = lr_classifier.run_logistic_regression_classifier_binary(
            run_label=run_label,
            run_id=run_id,
            run_results_dir=run_results_dir,
            X=X,
            y=Y,
            args_dict=args_dict
        )
    elif args_dict['algorithm'] == "linsvc":
        from algorithms import linsvc_classifier

        result, cv_results = linsvc_classifier.run_linear_svc_classifier_binary(
            run_label=run_label,
            run_id=run_id,
            run_results_dir=run_results_dir,
            X=X,
            y=Y,
            args_dict=args_dict
        )
    elif args_dict['algorithm'] == "qnn":
        if args_dict['q_backend'] == "statevector":
            from algorithms import qnn_classifier

            result,cv_results = qnn_classifier.run_qnn_simulator(
                run_label=run_label,
                run_id=run_id,
                run_results_dir=run_results_dir,
                X=X,
                y=Y,
                args_dict=args_dict
            )
    elif args_dict['algorithm'] == "qnn-reup":
        if args_dict['q_backend'] == "statevector":
            from algorithms import qnn_reup_classifier_simulator

            result,cv_results = qnn_reup_classifier_simulator.run_qnn_reup_simulator(
                run_label=run_label,
                run_id=run_id,
                run_results_dir=run_results_dir,
                X=X,
                y=Y,
                args_dict=args_dict
            )
        elif args_dict['q_backend'] == "cloud":
            from algorithms import qnn_reup_classifier_QPU

            result,cv_results = qnn_reup_classifier_QPU.run_qnn_reup_qpu(
                run_label=run_label,
                run_id=run_id,
                run_results_dir=run_results_dir,
                X=X,
                y=Y,
                random_state=args_dict['random_seed'],
                test_size=args_dict['test_split'],
                n_splits=args_dict['n_splits'],
                cv_method=args_dict['cv_method'],
                args_dict=args_dict
            )
    elif args_dict['algorithm'] == "vqc":
        if args_dict['q_backend'] == "statevector":
            from algorithms import vqc_classifier

            result,cv_results = vqc_classifier.run_vqc_simulator(
                run_label=run_label,
                run_id=run_id,
                run_results_dir=run_results_dir,
                X=X,
                y=Y,
                args_dict=args_dict
            )
    timings["train_seconds"] = time.perf_counter() - t_train_start
        
    # Class labels for this dataset (Benign=0, Malicious=1)
    CLASS_NAMES = ["Benign", "Malicious"]

    # Write cv_results dict to json file
    t_results_io_start = time.perf_counter()
    if cv_results is not None:
        cv_results_json_path = os.path.join(run_results_dir, "cv_results.json")
        with open(cv_results_json_path, "w") as json_file:
            json.dump(cv_results, json_file)
        mlflow.log_artifact(cv_results_json_path,artifact_path="cross_val_results")

        # Aggregated confusion matrix (sum across CV splits)
        tn_agg = fp_agg = fn_agg = tp_agg = 0
        try:
            tn_agg = int(np.sum(np.asarray(cv_results.get("true_negative", []), dtype=int)))
            fp_agg = int(np.sum(np.asarray(cv_results.get("false_positive", []), dtype=int)))
            fn_agg = int(np.sum(np.asarray(cv_results.get("false_negative", []), dtype=int)))
            tp_agg = int(np.sum(np.asarray(cv_results.get("true_positive", []), dtype=int)))
            cm_agg = np.array([[tn_agg, fp_agg], [fn_agg, tp_agg]], dtype=int)
            plot_confusion_matrix_from_counts(
                cm=cm_agg,
                run_results_dir=run_results_dir,
                filename="confusion_matrix_cv_agg.png",
                class_names=CLASS_NAMES,
            )
        except Exception as e:
            logger.warning(f"Failed to plot aggregated confusion matrix: {e}")

    # Log classifier results
    if args_dict['n_splits'] == 1:
        results_json_path = os.path.join(run_results_dir, "results.json")
        with open(results_json_path, "w") as json_file:
            json.dump(result, json_file)
        mlflow.log_artifact(results_json_path,artifact_path="single_run_results")

        keys_to_remove = [key for key, value in result.items() if isinstance(value, list)]
        for key in keys_to_remove:
            result.pop(key)
        mlflow.log_metrics(result)
    else:
        mlflow.log_metrics(result)

    # Always persist aggregated/summary metrics to the run folder
    summary_metrics_path = os.path.join(run_results_dir, "summary_metrics.json")
    with open(summary_metrics_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    # ── all_metrics.json: one comprehensive file with every metric ────────────
    # Builds per-fold breakdown from cv_results and combines with the aggregated
    # summary so there is a single JSON to read for analysis / the grid search.
    try:
        per_fold = {}
        cm_agg_counts = {"true_negative": 0, "false_positive": 0,
                         "false_negative": 0, "true_positive": 0}

        if cv_results is not None:
            # Identify scalar-per-fold keys (exclude nested lists like train_loss)
            scalar_keys = [
                k for k, v in cv_results.items()
                if isinstance(v, list) and v and not isinstance(v[0], (list, dict))
            ]
            n_folds = max((len(cv_results[k]) for k in scalar_keys), default=0)
            for fold_idx in range(n_folds):
                fold_data = {}
                for k in scalar_keys:
                    if fold_idx < len(cv_results[k]):
                        val = cv_results[k][fold_idx]
                        fold_data[k] = float(val) if isinstance(val, (np.floating, np.integer)) else val
                per_fold[f"fold_{fold_idx + 1}"] = fold_data

            cm_agg_counts = {
                "true_negative":  int(np.sum(np.asarray(cv_results.get("true_negative",  []), dtype=int))),
                "false_positive": int(np.sum(np.asarray(cv_results.get("false_positive", []), dtype=int))),
                "false_negative": int(np.sum(np.asarray(cv_results.get("false_negative", []), dtype=int))),
                "true_positive":  int(np.sum(np.asarray(cv_results.get("true_positive",  []), dtype=int))),
            }

        # Determine which CM image files were produced
        cm_files = sorted([
            f for f in os.listdir(run_results_dir)
            if "confusion_matrix" in f and f.endswith(".png")
        ])

        all_metrics = {
            "run_id":    run_id,
            "run_label": run_label,
            "class_names": CLASS_NAMES,
            "summary":   {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                          for k, v in result.items()},
            "per_fold":  per_fold,
            "confusion_matrix_aggregated": cm_agg_counts,
            "confusion_matrix_files": cm_files,
        }

        all_metrics_path = os.path.join(run_results_dir, "all_metrics.json")
        with open(all_metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2, default=str)
        mlflow.log_artifact(all_metrics_path, artifact_path="metrics")
    except Exception as e:
        logger.warning(f"Failed to write all_metrics.json: {e}")
    
    # Save images
    png_files = glob.glob(f'{run_results_dir}/*.png')
    for png in png_files:
        mlflow.log_artifact(png)
    timings["results_io_and_artifacts_seconds"] = time.perf_counter() - t_results_io_start

    # Log timings (as metrics + an artifact for full detail)
    timings["total_seconds"] = time.perf_counter() - t_script_start
    logger.info(
        "Timing summary (seconds): "
        + ", ".join([f"{k}={round(v, 4)}" for k, v in timings.items()])
    )
    for k, v in timings.items():
        try:
            mlflow.log_metric(k, float(v))
        except Exception:
            # Don't fail the run if MLflow rejects a metric for any reason
            pass
    timings_json_path = os.path.join(run_results_dir, "timings.json")
    with open(timings_json_path, "w") as f:
        json.dump(timings, f, indent=2)
    mlflow.log_artifact(timings_json_path, artifact_path="timing")

    # Optionally copy the processed dataset file into the run folder for provenance
    if args_dict.get("copy_processed_data_to_results", True) and processed_file_path and os.path.exists(processed_file_path):
        try:
            processed_copy_path = os.path.join(run_results_dir, "processed_dataset.csv")
            shutil.copy2(processed_file_path, processed_copy_path)
        except Exception as e:
            logger.warning(f"Failed to copy processed dataset into results dir: {e}")

    # Optionally copy the raw input dataset (e.g., the specific CSV subset used)
    raw_dataset_path = args_dict.get("dataset_path")
    if args_dict.get("copy_raw_dataset_to_results", True) and raw_dataset_path:
        try:
            # store as input_dataset.csv to keep naming consistent across runs
            raw_copy_path = os.path.join(run_results_dir, "input_dataset.csv")
            shutil.copy2(raw_dataset_path, raw_copy_path)
        except Exception as e:
            logger.warning(f"Failed to copy raw input dataset into results dir: {e}")

    # Create one consolidated bundle for convenience
    run_bundle_path = os.path.join(run_results_dir, "run_bundle.json")
    run_bundle = {
        "run_id": run_id,
        "run_label": run_label,
        "params": args_dict,
        "dataset_metadata": dataset_metadata,
        "summary_metrics": result,
        "timings_seconds": timings,
        "artifacts": {
            "run_params_json": os.path.basename(run_params_path),
            "dataset_metadata_json": os.path.basename(dataset_metadata_snapshot_path),
            "summary_metrics_json": os.path.basename(summary_metrics_path),
            "timings_json": os.path.basename(timings_json_path),
            "cv_results_json": "cv_results.json" if cv_results is not None else None,
            "processed_dataset_csv": "processed_dataset.csv" if args_dict.get("copy_processed_data_to_results", True) else None,
            "input_dataset_csv": "input_dataset.csv" if (args_dict.get("copy_raw_dataset_to_results", True) and args_dict.get("dataset_path")) else None,
        },
    }
    with open(run_bundle_path, "w") as f:
        json.dump(run_bundle, f, indent=2, default=str)

    mlflow.end_run()
    # Keep results by default (user requested persistence). Set --keep_results false to delete.
    if not args_dict.get("keep_results", True):
        shutil.rmtree(run_results_dir)
    
    print(f"{args_dict['algorithm']} classification complete")
    if cv_results is not None:
        print(f"F1 Score CV Results: {cv_results['f1']}")
        print(f"Median F1 Score: {round(np.median(cv_results['f1']),2)}")
    else:
        print(f"Balanced Accuracy: {round(result['balanced_acc']*100,2)}%")
