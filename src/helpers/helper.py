import argparse
import json
import os
import logging
import random
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

logger = logging.getLogger('azure')
logger.setLevel(logging.INFO)

def str2bool(v):
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Must specify true or false.')

def int_or_none(value):
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid int value: '{value}'")

# Some function to make the reup loop work
def sample_fx(array, y):
    """
    Distributes all items from an array into y groups as evenly as possible.

    :param array: List from which items will be distributed.
    :param y: Number of groups.
    :return: A list containing y groups, each of which is a list of items.
    """
    total_items = len(array)
    n = total_items // y  # Number of items in most groups
    remainder = total_items % y  # Extra items that don't fit evenly into groups

    results = []
    start_index = 0

    for i in range(y):
        # Add one extra item to the first 'remainder' groups
        end_index = start_index + n + (1 if i < remainder else 0)
        results.append(array[start_index:end_index])
        start_index = end_index

    return results

def pair_inputs_and_weights_single_qubit(inputs, weights):
    """
    Pairs input parameters with two consecutive weight parameters, cycling through inputs if there are more weights than inputs.

    :param inputs: List of input parameter names.
    :param weights: List of weight parameter names.
    :return: A list of pairs, each containing one input and two corresponding weights.
    """
    output = []
    input_length = len(inputs)
    
    for i in range(0, len(weights), 2):
        input_index = (i // 2) % input_length  # Cycle through input indices
        pair = [inputs[input_index], weights[i], weights[i+1]]
        output.append(pair)

    return output


def pair_inputs_and_weights_multi_qubit(inputs, weights, q_num_layers, q_num_qubit, q_reup_method):
    """
    Pairs input parameters with weight parameters in equal groups of q_num_qubit, 
    cycling through inputs if there are more weights than inputs and adding the qubit index.

    :param inputs: List of input parameter names.
    :param weights: List of weight parameter names.
    :param q_num_qubit: Number of qubits per group.
    :return: A nested list of pairs, each containing one input, a corresponding group of weights, and the qubit index.
    """

    logger.info(f"Reup Method: {q_reup_method}")

    if q_reup_method == "symmetrical":
        output = []
        num_weights = len(weights)
        num_inputs = len(inputs)
        num_layers = q_num_layers
        weight_index = 0
        for layer in range(num_layers):
            for input_index in range(num_inputs):
                input_output = []
                for qubit_index in range(q_num_qubit):
                    if weight_index < num_weights:
                        pair = [inputs[input_index]] + weights[weight_index:weight_index + 2] + [qubit_index]
                        weight_index += 2
                        input_output.append(pair)
                output.append(input_output)

    elif q_reup_method == "asymmetrical":
        output = []
        num_weights = len(weights)
        num_inputs = len(inputs)
        num_layers = q_num_layers
        num_pairs = num_inputs // q_num_qubit

        weight_index = 0
        for layer in range(num_layers):
            input_index = 0
            for pair in range(num_pairs):
                input_output = []
                for qubit_index in range(q_num_qubit):
                    pair = [inputs[input_index]] + weights[weight_index:weight_index + 2] + [qubit_index]
                    weight_index += 2
                    input_index += 1
                    input_output.append(pair)
                output.append(input_output)
    else:
        raise Exception(f"q_reup_method is: {q_reup_method}")
                
    return output




# parity maps bitstrings to 0 or 1
def parity(x):
    return "{:b}".format(x).count("1") % 2


def get_quantum_proba(quasi_dists):
    class_proba = []
    for quasi_dist in quasi_dists:
        positive_proba = 0.0

        for k, v in quasi_dist.items():
            result_parity = parity(k)
            if result_parity == 1:
                positive_proba += v

        negative_proba = 1.0 - positive_proba
        class_proba.append([negative_proba, positive_proba])

    return class_proba


def get_performance_metrics(test_labels, preds):
    performance_metrics = {}

    performance_metrics['accuracy'] = accuracy_score(test_labels, preds)
    performance_metrics['precision'] = precision_score(test_labels, preds)
    performance_metrics['recall'] = recall_score(test_labels, preds)
    performance_metrics['f1_score'] = f1_score(test_labels, preds)

    return performance_metrics


class RandomDisjointSplit:
    """
    Custom splitter that produces `n_splits` independent train/test partitions
    with exact sample counts and guaranteed no overlap.

    Each iteration:
      1. Randomly shuffle all indices with a per-split seed derived from
         `random_state` so every split sees a different sample ordering.
      2. Take the first `train_size` indices as the training set.
      3. Take the next `test_size` indices (from the remainder) as the test set.

    This gives `train_size` + `test_size` <= n_samples with zero overlap.
    """

    def __init__(self, n_splits=3, train_size=200, test_size=2000, random_state=None):
        self.n_splits     = n_splits
        self.train_size   = int(train_size)
        self.test_size    = int(test_size)
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        n = len(X)
        needed = self.train_size + self.test_size
        if needed > n:
            raise ValueError(
                f"RandomDisjointSplit: train_size ({self.train_size}) + "
                f"test_size ({self.test_size}) = {needed} exceeds dataset "
                f"size ({n}). Reduce train_size / test_size or increase the "
                f"dataset (e.g. raise max_samples_per_class)."
            )
        base_seed = self.random_state if self.random_state is not None else 42
        for i in range(self.n_splits):
            rng       = np.random.RandomState(base_seed + i)
            shuffled  = rng.permutation(n)
            train_idx = shuffled[:self.train_size]
            test_idx  = shuffled[self.train_size : self.train_size + self.test_size]
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def get_cross_val_obj(cv_method=None, n_splits=None, random_state=None, test_size=None,
                      train_count=None, test_count=None):
    """
    Return a CV splitter.

    When `train_count` and `test_count` are both provided (integers), returns a
    `RandomDisjointSplit` that draws exactly those many samples per iteration with
    no overlap between train and test.  This takes priority over `cv_method`.

    Otherwise falls back to the standard sklearn splitters:
      cv_method="skf"  → StratifiedKFold
      cv_method="sss"  → StratifiedShuffleSplit
    """
    if train_count is not None and test_count is not None:
        logging.info(
            f"* Setting up RandomDisjointSplit: "
            f"train={train_count}, test={test_count}, n_splits={n_splits}"
        )
        return RandomDisjointSplit(
            n_splits=n_splits or 3,
            train_size=int(train_count),
            test_size=int(test_count),
            random_state=random_state,
        )

    if cv_method == "skf":
        logging.info(f"* Setting up StratifiedKFold with {n_splits} splits")
        cv_split_obj = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state)
    elif cv_method == "sss":
        logging.info(f"* Setting up StratifiedShuffleSplit with {n_splits} splits")
        cv_split_obj = StratifiedShuffleSplit(
            n_splits=n_splits,
            test_size=test_size,
            random_state=random_state)
    else:
        raise Exception("* Must select valid cv method")

    return cv_split_obj
