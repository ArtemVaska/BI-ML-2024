import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    accuracy, precision, recall, f1 - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    precision_score = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall_score = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1_score = (
        2 * precision_score * recall_score / (precision_score + recall_score)
        if (precision_score + recall_score) != 0
        else 0
    )
    accuracy_score = (tp + tn) / (tp + tn + fp + fn)

    return precision_score, recall_score, f1_score, accuracy_score


def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    accuracy = np.count_nonzero(y_pred == y_true) / len(y_true)

    return accuracy


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """

    r2 = 1 - np.sum(np.square(y_true - y_pred)) / np.sum(
        np.square(y_true - np.mean(y_true))
    )

    return r2


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    mse = 1 / len(y_true) * np.sum(np.square(y_true - y_pred))

    return mse


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    mae = 1 / len(y_true) * np.sum(np.abs(y_true - y_pred))

    return mae
