"""Binary-classification metric exercises."""
import numpy as np

def confusion_matrix_from_scratch(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    """Return (tn, fp, fn, tp) by counting binary label pairs."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional.")
    if len(y_true) != len(y_pred):
        raise ValueError("Length of y_true must equal length of y_pred.")
    if np.any(~np.isin(y_true, [0, 1])):
        raise ValueError("Values of y_true must be 0 or 1.")
    if np.any(~np.isin(y_pred, [0, 1])):
        raise ValueError("Values of y_pred must be 0 or 1.")
    tn, fp, fn, tp = 0, 0, 0, 0
    for i in range(len(y_true)):
        if y_true[i] == 0 and y_pred[i] == 0:
            tn += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            fp += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            fn += 1
        elif y_true[i] == 1 and y_pred[i] == 1:
            tp += 1

    return (tn, fp, fn, tp)



def accuracy(tn: int, fp: int, fn: int, tp: int) -> float:
    """Return the fraction of all predictions that are correct; handle a zero denominator."""
    if tn + tp + fn + fp == 0:
        raise ValueError("Prediction evaluations are all 0.")
    return (tp + tn) / (tp + tn + fp + fn)

def precision(tn: int, fp: int, fn: int, tp: int) -> float:
    """Return the fraction of predicted positives that are correct."""
    if tp + fp == 0:
        raise ValueError("No true positives or false positives exist.")
    return tp / (tp + fp)

def recall(tn: int, fp: int, fn: int, tp: int) -> float:
    """Return the fraction of actual positives that are detected."""
    if tp + fn == 0:
        raise ValueError("No true positives or false negatives exist.")
    return tp / (tp + fn)

def specificity(tn: int, fp: int, fn: int, tp: int) -> float:
    """Return the fraction of actual negatives that are correctly rejected."""
    if tn + fp == 0:
        raise ValueError("No true negatives or false positives exist.")
    return tn / (tn + fp)

def f1_score(tn: int, fp: int, fn: int, tp: int) -> float:
    """Return the harmonic mean of precision and recall, handling undefined cases."""
    denominator = 2 * tp + fp + fn
    if denominator == 0:
        raise ValueError("F1 is undefined without actual or predicted positives.")
    return 2 * tp / denominator

def roc_points(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stretch: return FPR, TPR, and thresholds across useful cutoffs."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    thresholds = np.concatenate(
        ([np.inf], np.sort(np.unique(y_prob))[::-1])
    )
    fpr_values = []
    tpr_values = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix_from_scratch(y_true, y_pred)
        fpr = fp / (fp + tn)
        tpr = tp / (tp + fn)
        fpr_values.append(fpr)
        tpr_values.append(tpr)

    return (np.asarray(fpr_values), np.asarray(tpr_values), np.asarray(thresholds),)

def _validate_probabilities(y_true: np.ndarray, y_prob: np.ndarray,) -> tuple[np.ndarray, np.ndarray]:
    y_true, y_prob = _validate_binary_inputs(y_true, y_prob)
    y_prob = y_prob.astype(float)
    if np.any(~np.isfinite(y_prob)) or np.any((y_prob < 0) | (y_prob > 1)):
        raise ValueError("Probabilities must be finite and between 0 and 1.")
    return y_true, y_prob
        
def precision_recall_points(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stretch: return precision, recall, and thresholds across useful cutoffs."""
    y_true, y_prob = _validate_probabilities(y_true, y_prob)
    if not np.any(y_true == 1):
        raise ValueError("Precision-recall curves require actual positives.")

    thresholds = np.concatenate(([np.inf], np.sort(np.unique(y_prob))[::-1]))
    precision_values = []
    recall_values = []
    for threshold in thresholds:
        predictions = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix_from_scratch(y_true, predictions)
        precision_values.append(1.0 if tp + fp == 0 else tp / (tp + fp))
        recall_values.append(tp / (tp + fn))
    return np.asarray(precision_values), np.asarray(recall_values), thresholds
