
import logging
import mlflow

import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from keras.backend import clear_session
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tqdm import tqdm

from helpers.helper import get_cross_val_obj
from helpers.metrics import compute_metrics
from preprocessing.methods import get_scaler


class LossHistory(Callback):
    def on_train_begin(self, logs=None):
        self.train_losses = []
        self.val_losses = []
    
    def on_epoch_end(self, epoch, logs=None):
        self.train_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

def build_nn_model(input_dim):
    # # Experiment 1 config
    # model = Sequential()
    # model.add(Input(shape=(input_dim,)))
    # model.add(Dense(64, activation='relu'))
    # # model.add(Dense(32, activation='relu'))
    # # model.add(Dense(16, activation='relu'))
    # model.add(Dense(1, activation='sigmoid'))
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # return model

    # Experiment 3 config
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


    return model


def run_nn_classifier_binary(
        run_label=None,
        run_id=None, 
        run_results_dir=None,
        X=None, 
        y=None, 
        args_dict=None):
    
    logging.info("- Run NN Classifier")

    random_state=args_dict['random_seed']
    test_size=args_dict['test_split']
    n_splits=args_dict['n_splits']
    cv_method=args_dict['cv_method']
    nn_epochs = args_dict['epochs']

    cv_results = None

    # Initialize scaler if scaling is enabled
    scaler = get_scaler(args_dict["feature_scaler"]) if args_dict["scale_features"] else None
    
    # Get cross validation object (RandomDisjointSplit when train_count + test_count are set)
    cv_split_obj = get_cross_val_obj(
        cv_method=cv_method,
        n_splits=n_splits,
        test_size=test_size,
        random_state=random_state,
        train_count=args_dict.get("train_count"),
        test_count=args_dict.get("test_count"),
    )

    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    cv_results = {}
    current_split = 0
    for train_index, test_index in tqdm(cv_split_obj.split(X, y), total=n_splits, desc="Running CV"):

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

        # Clear previous model from memory
        clear_session()

        # Set input dimension
        input_dim = X_train.shape[1]

        # Build ANN model
        model = build_nn_model(input_dim)
        model.summary()
        trainable_params = int(np.sum([K.count_params(w) for w in model.trainable_weights]))
        print(f"Number of trainable parameters: {trainable_params}")

        # TRAIN MODEL
        loss_history = LossHistory()
        model.fit(
            X_train, 
            y_train, 
            epochs=nn_epochs, 
            batch_size=32, 
            verbose=1, 
            validation_split=0.2,
            callbacks=[loss_history]
        )

        # PREDICT: Predict class label and probabilities on test dataset
        y_pred = (model.predict(X_test) > 0.5).astype(int)
        y_prob = model.predict(X_test)

        # EVALUATE: Compute metrics and plot graphs for current_split on test labels
        cv_result = compute_metrics(
            run_label=f"{run_label}_cv{current_split}", 
            run_id=run_id, 
            y_test=y_test, 
            y_pred=y_pred, 
            y_prob=y_prob,
            run_results_dir=run_results_dir)
        
        cv_result['train_loss'] = loss_history.train_losses
        cv_result['val_loss'] = loss_history.val_losses

        # Curate total cross-validation dictionary with each metric
        for k, v in cv_result.items():
            if k not in cv_results:
                cv_results[k] = []
            cv_results[k].append(v)

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
    
    result["nb_trainable_params"] = trainable_params

    return result, cv_results
    