import logging
import os
import json
from qiskit_ibm_runtime import QiskitRuntimeService


def init_quantum_environment(quantum_config_path=None):
    logging.info("* Initializing IBM Quantum Environment")

    if not os.path.exists(quantum_config_path):
        raise Exception("No configuration provided for IBM Quantum Service")

    ibm_quantum_token = None
    instance = None
    with open(quantum_config_path) as config_file:
        config = json.load(config_file)
        ibm_quantum_token = config["token"]
        instance = config["instance"]

    if ibm_quantum_token is None:
        raise Exception("No token provided for IBM Quantum Service")
    if instance is None:
        instance = "ibm-q/open/main"

    QiskitRuntimeService.save_account(
        channel="ibm_quantum",
        token=ibm_quantum_token,
        overwrite=True
    )

    return QiskitRuntimeService(instance=instance)