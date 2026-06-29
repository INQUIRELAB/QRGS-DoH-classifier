import logging

import mlflow
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from qiskit.circuit.library import EfficientSU2, RealAmplitudes, ZZFeatureMap
from qiskit.primitives import BackendSampler
from qiskit_aer import AerSimulator
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_ibm_runtime import Sampler
from qiskit_ibm_runtime.fake_provider import FakeBelem, FakeKyiv
from qiskit_machine_learning.algorithms.classifiers import VQC
from tqdm import tqdm

# from core.ibmcloud import init_quantum_environment
from helpers.helper import get_cross_val_obj
from helpers.metrics import compute_metrics
from preprocessing.methods import get_scaler


def callback_log_loss(weights, obj_func_eval):
    train_loss.append(obj_func_eval)
    mlflow.log_metric("qnn_loss", obj_func_eval)


def run_vqc_simulator(
        run_label=None,
        run_id=None,
        run_results_dir=None,
        X=None, 
        y=None,
        args_dict=None):
    
    logging.info("- Run VQC Classifier on Simulator")
    
    global train_loss

    random_state=args_dict['random_seed']
    test_size=args_dict['test_split']
    n_splits=args_dict['n_splits']
    cv_method=args_dict['cv_method']

    # Set up backend ---------------------------------------------------
    backend = AerSimulator(method=args_dict["q_backend"])
    if args_dict["q_noise"]:
        device_backend = FakeKyiv() # TODO: Test and switch to FakeKyiv()
        backend = backend.from_backend(device_backend)
    sampler = BackendSampler(backend=backend)


    # Set up circuit ---------------------------------------------------
    objective_func_vals = []
    num_inputs = X.shape[1]
    algorithm_globals.random_seed = args_dict["q_random_state"]
    output_shape = 2 # corresponds to the number of classes,

    # Feature Map (i.e., input layer)
    feature_map = ZZFeatureMap(feature_dimension=num_inputs, reps=args_dict["q_feature_map_reps"])

    # Ansatz (i.e., weights)
    ansatz = None
    if args_dict["q_ansatz"] == "realamp":
        ansatz = RealAmplitudes(num_qubits=num_inputs, reps=args_dict["q_ansatz_reps"])
    elif args_dict["q_ansatz"] == "su2":
        ansatz = EfficientSU2(num_qubits=num_inputs, reps=args_dict["q_ansatz_reps"])
    else:
        raise Exception("No Ansatz specified")

    # Optimizer
    if args_dict["q_optimizer"] == "cobyla":
        optimizer = COBYLA(
            maxiter=args_dict["epochs"]
        )
    elif args_dict["q_optimizer"] == "spsa":
        optimizer = SPSA(
            maxiter=args_dict["epochs"]
        )
    else:
        raise Exception("No optimizer specified")

    # Sampler
    sampler = Sampler(backend=backend)

    # Train and test ---------------------------------------------------
    
    cv_results = None

    # Initialize scaler if scaling is enabled
    scaler = get_scaler(args_dict["feature_scaler"]) if args_dict["scale_features"] else None

    # Get cross validation object
    cv_split_obj = get_cross_val_obj(cv_method=cv_method, n_splits=n_splits, test_size=test_size, random_state=random_state)

    cv_results = {}
    cv_results['train_loss'] = []
    current_split = 0
    for train_index, test_index in tqdm(cv_split_obj.split(X, y), total=len(range(n_splits)), desc="Running CV"):  # noqa: E501

        train_loss = []
        # Classifier
        classifier = VQC(
            sampler=sampler,
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=optimizer,
            callback=callback_log_loss,
        )

        current_split += 1

        # SPLIT: Apply split to X and y
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # DOWNSAMPLING: Apply only to training data
        if args_dict['downsample']:
            rus = RandomUnderSampler(sampling_strategy='auto', random_state=random_state)
            X_train, y_train = rus.fit_resample(X_train, y_train)

        # These values vary by CV split; log as metrics (params can't change).
        mlflow.log_metric("percent_pos_class_y_train", round(sum(y_train)/len(y_train), 4), step=current_split)
        mlflow.log_metric("length_y_train", float(len(y_train)), step=current_split)
        mlflow.log_metric("percent_pos_class_y_test", round(sum(y_test)/len(y_test), 4), step=current_split)
        mlflow.log_metric("length_y_test", float(len(y_test)), step=current_split)

        # FEATURE SCALING: Fit on train, transform both train/test
        if scaler:
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # TRAIN MODEL
        classifier.fit(X_train, y_train)

        # PREDICT: Predict class label and probabilities on test dataset
        y_pred = classifier.predict(X_test)

        # EVALUATE: Compute metrics and plot graphs for current_split on test labels
        cv_result = compute_metrics(
            run_label=f"{run_label}_cv{current_split}", 
            run_id=run_id, 
            y_test=y_test, 
            y_pred=y_pred, 
            y_prob=None,
            run_results_dir=run_results_dir)

        # Curate total cross-validation dictionary with each metric
        for k, v in cv_result.items():
            if k not in cv_results:
                cv_results[k] = []
            cv_results[k].append(v)

        cv_results['train_loss'].append(train_loss)

    # Aggregate cross-validation results
    result = {}
    mlflow_metrics = ["acc","f1","precision","recall","roc_auc","specificity","balanced_acc"]
    for k,v in cv_results.items():
        if k in mlflow_metrics:
            result[f"cv_mean_{k}"] = np.mean(cv_results[k])
            result[f"cv_SD_{k}"] = np.std(cv_results[k])
            result[f"cv_median_{k}"] = np.median(cv_results[k])
            result[f"cv_025_{k}"] = np.percentile(cv_results[k], 2.5)
            result[f"cv_975_{k}"] = np.percentile(cv_results[k], 97.5)

    return result, cv_results
