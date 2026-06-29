import logging
import os

import mlflow
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.circuit.library import EfficientSU2, RealAmplitudes, ZZFeatureMap
from qiskit.primitives import BackendEstimator, BackendSampler
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

from qiskit_algorithms.optimizers import COBYLA, SPSA, ADAM
from qiskit_algorithms.utils import algorithm_globals
from qiskit_ibm_runtime.fake_provider import FakeBelem,FakeKyiv

# https://docs.quantum.ibm.com/migration-guides/v2-primitives
# from qiskit_ibm_runtime import Estimator 
from qiskit_ibm_runtime import EstimatorV2

from qiskit_ibm_runtime import Options, Sampler, Session
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm

from core.ibmcloud import init_quantum_environment
from helpers.helper import (
    get_cross_val_obj, 
    sample_fx
)
from helpers.metrics import compute_metrics
from core.quantum_reup import create_quantum_reup_circuit

def mlflow_callback_log_loss(weights, obj_func_eval):
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
            fixed_pred = estimator_classifier.predict([X_test[ix]]).flatten()
            while fixed_pred[0] == 0:
                fixed_pred = estimator_classifier.predict([X_test[ix]]).flatten()
            y_pred[ix] = fixed_pred[0]
    return y_pred



def run_qnn_reup_qpu(
        run_label=None,
        run_id=None,
        run_results_dir=None,
        X=None, 
        y=None, 
        random_state=42,
        test_size=0.3,
        n_splits=1,
        cv_method=None,
        args_dict=None):

    # Optional set random seed for qiskit
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
    backend = None
    quantum_config_path = os.path.join(args_dict['parent_directory'],"config/quantum_config.json")

    service = init_quantum_environment(quantum_config_path)
    backend = service.least_busy(operational=True, simulator=False, min_num_qubits=127)

    qc = transpile(qc, backend, optimization_level=args_dict['q_optimization_level'])
    
    cv_results = None
    # with Session(service, backend=wandb.config["quantum_params"]["backend"]) as session:
    # with Session(service, backend=backend) as session:
    with Session(backend=backend) as session:

        estimator = EstimatorV2()

        observable = SparsePauliOp.from_list([("Z" * qc.num_qubits, 1)])

        # estimator = Estimator(session=session)
        # options = Options()
        # options.optimization_level = args_dict["q_optimization_level"]
        # options.execution.shots = args_dict["q_shots"]

        estimator_qnn = EstimatorQNN(
            circuit=qc, 
            estimator=estimator,
            observables=observable, 
            input_params=inputs, 
            weight_params=weights
        )
        estimator_circuit_path = os.path.join(run_results_dir, "qnnreup_estimator_circuit.png")
        estimator_qnn._circuit.decompose().draw(output='mpl', style="clifford",filename=estimator_circuit_path)

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
                maxiter=args_dict["epochs"]
            )
        else:
            raise Exception("No optimizer specified")

        # construct neural network classifier
        estimator_classifier = NeuralNetworkClassifier(
            estimator_qnn, 
            optimizer=optimizer, 
            # callback=mlflow_callback_log_loss
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args_dict['test_split'], random_state=random_state)

        # Fit model
        estimator_classifier.fit(X_train, y_train)

        # Make predictions
        y_pred = estimator_classifier.predict(X_test)

        # Find and fix any 'zero' predictions
        y_pred = fix_zero_preds(X_test, y_pred, estimator_classifier)

        # Compute performance metrics and plot graphs
        result = compute_metrics(
            run_label=run_label, 
            run_id=run_id,
            y_test=y_test, 
            y_pred=y_pred,
            y_prob=None,
            run_results_dir=run_results_dir)
        
        # TODO: add shape of X_train and others to result

        session.close()

        return result, cv_results


