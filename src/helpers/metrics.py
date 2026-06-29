import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             fbeta_score, matthews_corrcoef, precision_score,
                             recall_score, roc_curve, balanced_accuracy_score)
import os

def compute_metrics(
        run_label=None,
        run_id=None,
        y_test=None,
        y_pred=None,
        y_prob=None,
        run_results_dir=None,
        class_names=None):
    
    
    logging.info(f"* Computing classifier ({run_label}) performance metrics")

    # For QNN-REUP, labels are often encoded as {-1, +1} to match the natural
    # range of expectation values. For reporting (and consistency with most ML
    # tooling), remap {-1, +1} -> {0, 1}.
    y_test = np.asarray(y_test).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    uniq = set(np.unique(np.concatenate([y_test, y_pred])).tolist())
    if uniq.issubset({-1, 1}):
        y_test = np.where(y_test == -1, 0, 1)
        y_pred = np.where(y_pred == -1, 0, 1)
    
    result = {}
    result['acc'] = accuracy_score(y_test, y_pred)
    result["balanced_acc"] = balanced_accuracy_score(y_test, y_pred)
    result['precision'] = precision_score(y_test, y_pred, zero_division=0)
    result['recall'] = recall_score(y_test, y_pred, zero_division=0)
    result['f1'] = f1_score(y_test, y_pred, zero_division=0)
    result['f2'] = fbeta_score(y_test, y_pred, beta=2, zero_division=0)
    result['mcc'] = matthews_corrcoef(y_test, y_pred)

    conf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = map(int, conf_matrix.ravel())

    result['true_negative'] = tn
    result['false_positive'] = fp
    result['false_negative'] = fn
    result['true_positive'] = tp
    result['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    plot_confusion_matrix(
        run_label=run_label,
        run_id=run_id,
        y_true=y_test,
        y_pred=y_pred,
        normalize=False,
        cmap=plt.cm.Blues,
        run_results_dir=run_results_dir,
        class_names=class_names,
    )

    if y_prob is not None:
        
        # Compute and store additional metrics
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        result['roc_auc'] = auc(fpr, tpr)
        result['fpr'] = fpr.tolist()
        result['tpr'] = tpr.tolist()
        result['roc_thresholds'] = thresholds.tolist()
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
        result['calibration_curve_prob_true'] = prob_true.tolist()
        result['calibration_curve_prob_pred'] = prob_pred.tolist()

        plot_roc_curve(
            fpr,
            tpr,
            result['roc_auc'],
            run_label,
            run_id,
            result['acc'],
            result['precision'],
            result['recall'],
            result['specificity'],
            result['f1'],
            run_results_dir
        )

        plot_calibration_curve(run_label, run_id, prob_pred, prob_true,run_results_dir)

    return result


def plot_feature_selection(
        run_label=None,
        run_id=None,
        features=None, 
        selected_features=None):
    
    fig = plt.figure(figsize=(16, 6))
    ax = fig.add_axes([0.1, 0.3, .9, .7])
    ax.set_ylabel('Number of Selected Features')
    ax.set_xticks(np.arange(len(features)))
    ax.set_xticklabels(features, rotation=90)
    ax.set_yticks(np.arange(len(features)))
    ax.set_yticklabels(np.arange(1, len(features)+1))
    # Set a grid on minor ticks
    ax.set_xticks(np.arange(-0.5, len(features)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(features)), minor=True)
    ax.grid(which='minor', color='black')
    ax.imshow(selected_features, cmap=colors.ListedColormap(['white', 'red']))

    fx_select_path = os.path.join(run_label, run_id,"feature_selection.png")
    fig.savefig(fx_select_path, dpi=600)
    fig.clear()


def plot_roc_curve(
        fpr,
        tpr,
        roc_auc,
        run_label,
        run_id,
        acc,
        precision,
        recall,
        specificity,
        f1,
        run_results_dir):

    # Plot ROC Curve
    # plt.figure()
    plt.plot(
        fpr,
        tpr,
        color='darkorange',
        lw=2,
        label=f'ROC curve (AUC = {roc_auc:.2f})'
    )

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    # Annotate ROC plot with metric values
    metrics_text = f"""
        Accuracy: {acc:.2f}\n
        Precision: {precision:.2f}\n
        Sensitivity: {recall:.2f}\n
        Specificity: {specificity:.2f}\n
        F1 Score: {f1:.2f}
        """
    plt.annotate(
            metrics_text,
            xy=(0.6, 0.3),
            xycoords='axes fraction',
            bbox=dict(boxstyle="round, pad=0.5", fc="white", ec="black", lw=2)
        )

    plt.legend(loc="lower right")

    fig_path = os.path.join(run_results_dir, f"{run_label}_roc_curve.png")
    plt.savefig(fig_path, dpi=600)
    plt.clf()


def plot_calibration_curve(run_label, run_id, prob_pred, prob_true,run_results_dir):
    # plt.figure()
    plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Model')
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle='--',
        color='red',
        label='Perfectly calibrated'
    )
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.legend()

    fig_path = os.path.join(run_results_dir, f"{run_label}_calibration_curve.png")
    plt.savefig(
        fig_path,
        dpi=600)
    plt.clf()


def plot_confusion_matrix(
        run_label,
        run_id,
        y_true,
        y_pred,
        normalize=False,
        cmap=plt.cm.Blues,
        run_results_dir=None,
        class_names=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    `class_names` is an optional list of two strings for the tick labels,
    e.g. ["Benign", "Malicious"]. Falls back to numeric class indices.
    """
    # Compute confusion matrix
    # Infer class labels from the data, with a consistent axis order.
    y_true_arr = np.asarray(y_true).reshape(-1)
    y_pred_arr = np.asarray(y_pred).reshape(-1)

    # If labels are {-1, +1}, remap to {0, 1} for plotting.
    uniq = set(np.unique(np.concatenate([y_true_arr, y_pred_arr])).tolist())
    if uniq.issubset({-1, 1}):
        y_true_arr = np.where(y_true_arr == -1, 0, 1)
        y_pred_arr = np.where(y_pred_arr == -1, 0, 1)
        uniq = {0, 1}

    # For binary classification, always plot in [0, 1] order.
    if uniq.issubset({0, 1}):
        numeric_classes = [0, 1]
    else:
        numeric_classes = sorted(np.unique(np.concatenate([y_true_arr, y_pred_arr])).tolist())

    tick_labels = class_names if (class_names and len(class_names) == len(numeric_classes)) else numeric_classes

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=numeric_classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=tick_labels, yticklabels=tick_labels,
           ylabel='True label',
           xlabel='Predicted label')

    # Put x-axis (predicted) tick labels and label on top.
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.tick_params(axis='x', top=True, bottom=False, labeltop=True, labelbottom=False)

    # Do not rotate tick labels (keep them horizontal).
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    fig_path = os.path.join(run_results_dir, f"{run_label}_confusion_matrix.png")
    fig.savefig(fig_path, dpi=600)
    plt.close(fig)


def plot_confusion_matrix_from_counts(
        cm,
        run_results_dir,
        filename="confusion_matrix_cv_agg.png",
        cmap=plt.cm.Blues,
        class_names=None):
    """
    Plot a confusion matrix from raw counts.

    Parameters
    - cm          : array-like shape (2,2) interpreted as [[TN, FP], [FN, TP]].
    - class_names : optional list of two strings, e.g. ["Benign", "Malicious"].
                    Falls back to [0, 1] when not provided.
    - Saves into run_results_dir/filename.
    """
    cm_arr = np.asarray(cm, dtype=int)
    if cm_arr.shape != (2, 2):
        raise ValueError(f"cm must be shape (2,2), got {cm_arr.shape}")

    tick_labels = class_names if (class_names and len(class_names) == 2) else [0, 1]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_arr, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        ylabel='True label',
        xlabel='Predicted label'
    )

    # Put x-axis (predicted) tick labels and label on top.
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.tick_params(axis='x', top=True, bottom=False, labeltop=True, labelbottom=False)

    # Keep tick labels horizontal.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    thresh = cm_arr.max() / 2.0 if cm_arr.size else 0
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{int(cm_arr[i, j])}",
                ha="center", va="center",
                color="white" if cm_arr[i, j] > thresh else "black"
            )

    fig.tight_layout()
    os.makedirs(run_results_dir, exist_ok=True)
    fig_path = os.path.join(run_results_dir, filename)
    fig.savefig(fig_path, dpi=600)
    plt.close(fig)
