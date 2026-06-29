
import os
import logging
import time

import mlflow
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from qiskit import transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_algorithms.optimizers import (
    ADAM,
    COBYLA,
    SPSA,
    L_BFGS_B,
    NELDER_MEAD,
    SLSQP,
    POWELL,
    TNC,
)
from qiskit_algorithms.utils import algorithm_globals
from qiskit_ibm_runtime.fake_provider import FakeKyiv
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
from tqdm import tqdm

from core.quantum_reup import create_quantum_reup_circuit
from helpers.helper import get_cross_val_obj
from helpers.metrics import compute_metrics
from preprocessing.methods import get_scaler


def mlflow_callback_log_loss(weights, obj_func_eval):
    train_loss.append(obj_func_eval)
    mlflow.log_metric("qnn_reup_loss", obj_func_eval)


def fix_zero_preds(X_test, y_pred, estimator_classifier):
    """
    Find all predictions in y_pred that are zero (0) and re-run predict() until
    a non-zero answer is returned. Then replace that answer in the original y_pred. 
    
    Args:
        X_test (arr): samples to do prediction on 
        y_pred (arr): predictions 
        estimator_classifier (obj): classifier object

    Returns:
        y_pred
    """
    # Iterate through each index in y_pred
    for ix in range(len(y_pred)):
        if y_pred[ix] == 0.:
            print(ix)
            nb_zero_preds.append(ix)
            fixed_pred = estimator_classifier.predict([X_test[ix]]).flatten()
            while fixed_pred[0] == 0:
                fixed_pred = estimator_classifier.predict([X_test[ix]]).flatten()
            y_pred[ix] = fixed_pred[0]
    return y_pred


# This function was ported over from main_reup.ipynb
def run_qnn_reup_simulator(
        run_label=None,
        run_id=None,
        run_results_dir=None,
        X=None, 
        y=None, 
        args_dict=None):
    
    logging.info("- Run QNN-REUP Classifier on Simulator")
    
    global train_loss
    global nb_zero_preds

    random_state=args_dict['random_seed']
    test_size=args_dict['test_split']
    n_splits=args_dict['n_splits']
    cv_method=args_dict['cv_method']

    algorithm_globals.random_seed = args_dict["q_random_state"]

    # Set number of inputs (i.e., number of features)
    num_inputs = X.shape[1]
    inputs = ParameterVector("input", num_inputs)
    
    # calculate number of weights in array based on num_layers and num_qubit
    # Multiply by two because there are 2 parameters per input
    if (args_dict['q_num_qubit'] > 1) and (args_dict['q_reup_method'] == "asymmetrical"):
        num_weights = (num_inputs*2) * args_dict['q_num_layers']
    else:
        num_weights = (num_inputs*2) * args_dict['q_num_layers'] * args_dict["q_num_qubit"]
    weights = ParameterVector("weight", num_weights)
    trainable_params = len(weights)

    qc = create_quantum_reup_circuit(
        num_inputs=num_inputs,
        inputs=inputs,
        weights=weights,
        args_dict=args_dict)
    circuit_diagram_path = os.path.join(run_results_dir, "qnnreup_circuit_circuit.png")
    qc.draw(output='mpl', style="clifford",filename=circuit_diagram_path)

    # =-=-=-= Begin Backend Config =-=-=-=-=-=-=-=-=-=-=

    backend = AerSimulator(method=args_dict["q_backend"])

    qc = transpile(qc, backend, optimization_level=args_dict['q_optimization_level']) # add transpile seed?

    estimator_qnn = EstimatorQNN(
        circuit=qc, 
        input_params=inputs, 
        weight_params=weights
    )
    
    optimizer = None
    if args_dict["q_optimizer"] == "cobyla":
        optimizer = COBYLA(
            maxiter=args_dict["epochs"]
        )
    elif args_dict["q_optimizer"] == "spsa":
        optimizer = SPSA(
            maxiter=args_dict["epochs"]
        )
    elif args_dict["q_optimizer"] == "adam":
        optimizer = ADAM(
            maxiter=args_dict["epochs"],
            lr=0.001
        )
    elif args_dict["q_optimizer"] in ("l-bfgs-b", "lbfgsb", "l_bfgs_b"):
        optimizer = L_BFGS_B(maxiter=args_dict["epochs"])
    elif args_dict["q_optimizer"] in ("nelder-mead", "nelder_mead"):
        optimizer = NELDER_MEAD(maxiter=args_dict["epochs"])
    elif args_dict["q_optimizer"] in ("slsqp",):
        optimizer = SLSQP(maxiter=args_dict["epochs"])
    elif args_dict["q_optimizer"] in ("powell",):
        optimizer = POWELL(maxiter=args_dict["epochs"])
    elif args_dict["q_optimizer"] in ("tnc",):
        optimizer = TNC(maxiter=args_dict["epochs"])
    else:
        raise Exception("No optimizer specified")


    cv_results = None

    # Class labels used in confusion matrix plots
    class_names = args_dict.get("class_names", ["Benign", "Malicious"])

    # Initialize scaler if scaling is enabled
    scaler = get_scaler(args_dict["feature_scaler"]) if args_dict["scale_features"] else None

    # Get cross validation object.
    # When train_count + test_count are set, RandomDisjointSplit is used:
    # each iteration draws exactly train_count random training rows and
    # test_count random test rows from the remaining samples (no overlap).
    cv_split_obj = get_cross_val_obj(
        cv_method=cv_method,
        n_splits=n_splits,
        test_size=test_size,
        random_state=random_state,
        train_count=args_dict.get("train_count"),
        test_count=args_dict.get("test_count"),
    )

    cv_results = {}
    cv_results['nb_zero_preds'] = []
    cv_results['train_loss'] = []
    current_split = 0
    for train_index, test_index in tqdm(cv_split_obj.split(X, y), total=n_splits, desc="Running CV"):

        nb_zero_preds = []
        train_loss = []
        # construct neural network classifier
        estimator_classifier = NeuralNetworkClassifier(
            estimator_qnn,
            optimizer=optimizer,
            callback=mlflow_callback_log_loss
        )

        current_split += 1

        # SPLIT: Apply split to X and y
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # DOWNSAMPLING: Apply only to training data
        if args_dict['downsample']:
            rus = RandomUnderSampler(sampling_strategy='auto', random_state=random_state)
            X_train, y_train = rus.fit_resample(X_train, y_train)

        percent_positive_class_ytrain = round(np.count_nonzero(y_train == 1)/len(y_train), 4)
        percent_positive_class_ytest  = round(np.count_nonzero(y_test  == 1)/len(y_test),  4)

        mlflow.log_metric("percent_pos_class_y_train", percent_positive_class_ytrain, step=current_split)
        mlflow.log_metric("length_y_train",            float(len(y_train)),           step=current_split)
        mlflow.log_metric("percent_pos_class_y_test",  percent_positive_class_ytest,  step=current_split)
        mlflow.log_metric("length_y_test",             float(len(y_test)),            step=current_split)

        # FEATURE SCALING: Fit on train, transform both train/test
        if scaler:
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)

        # TRAIN MODEL — timed
        t_fold_start  = time.perf_counter()
        t_train_start = time.perf_counter()
        estimator_classifier.fit(X_train, y_train)
        fold_train_seconds = time.perf_counter() - t_train_start

        # PREDICT
        y_pred = estimator_classifier.predict(X_test)
        y_pred = fix_zero_preds(X_test, y_pred, estimator_classifier)
        fold_total_seconds = time.perf_counter() - t_fold_start

        mlflow.log_metric("fold_train_seconds", fold_train_seconds, step=current_split)
        mlflow.log_metric("fold_total_seconds", fold_total_seconds, step=current_split)
        logging.info(
            f"  CV {current_split}: train={fold_train_seconds:.1f}s  "
            f"fold_total={fold_total_seconds:.1f}s"
        )

        # EVALUATE: compute all metrics + save per-fold confusion matrix
        cv_result = compute_metrics(
            run_label=f"{run_label}_cv{current_split}",
            run_id=run_id,
            y_test=y_test,
            y_pred=y_pred,
            y_prob=None,
            run_results_dir=run_results_dir,
            class_names=class_names,
        )

        # Accumulate per-fold metrics
        for k, v in cv_result.items():
            if k not in cv_results:
                cv_results[k] = []
            cv_results[k].append(v)

        cv_results['train_loss'].append(train_loss)
        cv_results['nb_zero_preds'].append(len(nb_zero_preds))
        cv_results.setdefault('fold_train_seconds', []).append(round(fold_train_seconds, 3))
        cv_results.setdefault('fold_total_seconds', []).append(round(fold_total_seconds, 3))

    # ── Aggregate cross-validation results ────────────────────────────────────
    result = {}
    result["nb_trainable_params"] = trainable_params

    # Scalar performance metrics: compute mean, SD, median, and 95% CI
    scalar_metrics = [
        "acc", "balanced_acc", "f1", "f2", "mcc",
        "precision", "recall", "specificity", "roc_auc",
    ]
    for k in scalar_metrics:
        if k in cv_results and cv_results[k]:
            vals = cv_results[k]
            result[f"cv_mean_{k}"]   = float(np.mean(vals))
            result[f"cv_SD_{k}"]     = float(np.std(vals))
            result[f"cv_median_{k}"] = float(np.median(vals))
            result[f"cv_025_{k}"]    = float(np.percentile(vals, 2.5))
            result[f"cv_975_{k}"]    = float(np.percentile(vals, 97.5))

    # Confusion matrix counts: aggregate by summing across folds
    for cm_key in ("true_negative", "false_positive", "false_negative", "true_positive"):
        if cm_key in cv_results and cv_results[cm_key]:
            result[f"agg_{cm_key}"] = int(np.sum(cv_results[cm_key]))

    # Timing: per-fold train time and full fold time
    for t_key in ("fold_train_seconds", "fold_total_seconds"):
        if t_key in cv_results and cv_results[t_key]:
            vals = cv_results[t_key]
            result[f"mean_{t_key}"]   = round(float(np.mean(vals)),   3)
            result[f"median_{t_key}"] = round(float(np.median(vals)),  3)
            result[f"total_{t_key}"]  = round(float(np.sum(vals)),    3)

    return result, cv_results

