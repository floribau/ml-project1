"""Evaluation metrics"""

import numpy as np


def _prepare_labels(y_true, y_pred):
    """Convert label inputs to one-dimensional arrays and validate their shape."""
    true = np.asarray(y_true).ravel()
    pred = np.asarray(y_pred).ravel()

    if true.size == 0:
        raise ValueError("y_true and y_pred must not be empty")
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same number of samples")

    return true, pred


def confusion_counts(y_true, y_pred, pos_label=1):
    """Return binary confusion-matrix counts as ``(true_positive, false_positive,
    true_negative, false_negative)``.

    Labels may be encoded as ``0/1`` or ``-1/1``. Any label other than
    ``positive_label`` is treated as the negative class.
    """
    true, pred = _prepare_labels(y_true, y_pred)

    true_positive = np.count_nonzero((true == pos_label) & (pred == pos_label))
    false_positive = np.count_nonzero((true != pos_label) & (pred == pos_label))
    true_negative = np.count_nonzero((true != pos_label) & (pred != pos_label))
    false_negative = np.count_nonzero((true == pos_label) & (pred != pos_label))

    return true_positive, false_positive, true_negative, false_negative


def accuracy_from_predictions(y_true, y_pred):
    """Calculate the fraction of correctly classified samples."""
    true, pred = _prepare_labels(y_true, y_pred)
    return np.count_nonzero(true == pred) / true.size


def accuracy_from_model(y_true, X, w):
    """compute the accuracy on the given dataset (X, y) using the model w.

    Args:
        y_true: the corresponding +1 or -1 labels, shape = (num_examples)
        X: the full dataset matrix, shape = (num_examples, num_features)
        w: shape = (num_features)

    Returns:
        scalar, between 0 and 1
    """
    pred = X @ w
    correct_prediction = y_true * pred  # prediction is correct if the sign is positive.
    return correct_prediction[correct_prediction >= 0].shape[0] / y_true.shape[0]


def precision(y_true, y_pred, pos_label=1):
    """Calculate the positive predictive value for a binary classifier."""
    true_positive, false_positive, _, _ = confusion_counts(
        y_true, y_pred, pos_label
    )
    denominator = true_positive + false_positive
    return true_positive / denominator if denominator else 0.0


def recall(y_true, y_pred, pos_label=1):
    """Calculate the true-positive rate for a binary classifier."""
    true_positive, _, _, false_negative = confusion_counts(
        y_true, y_pred, pos_label
    )
    denominator = true_positive + false_negative
    return true_positive / denominator if denominator else 0.0


def f1_score(y_true, y_pred, pos_label=1):
    """Calculate the harmonic mean of precision and recall."""
    metric_precision = precision(y_true, y_pred, pos_label)
    metric_recall = recall(y_true, y_pred, pos_label)
    denominator = metric_precision + metric_recall
    return (
        2 * metric_precision * metric_recall / denominator
        if denominator
        else 0.0
    )
