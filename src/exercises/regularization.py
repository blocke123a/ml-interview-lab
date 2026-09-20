"""Regularization and bias-variance exercises."""
import numpy as np

def l1_penalty(weights: np.ndarray) -> float:
    """Return the sum of absolute coefficient values, excluding intercepts when appropriate."""
    return float(np.sum(np.abs(np.asarray(weights, dtype=float))))

def l2_penalty(weights: np.ndarray) -> float:
    """Return the sum of squared coefficient values, excluding intercepts when appropriate."""
    weights = np.asarray(weights, dtype=float)
    return float(np.sum(weights ** 2))

def regularized_mse(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray, alpha: float, penalty: str) -> float:
    """Combine MSE with L1 or L2; choose and document a consistent scaling convention."""
    if alpha < 0:
        raise ValueError("Alpha cannot be negative.")
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("Targets and predictions must have the same nonzero shape.")

    mse = float(np.mean((y_true - y_pred)**2))
    penalty_name = penalty.lower()
    if penalty_name == "l1":
        penalty_value = l1_penalty(weights)
    elif penalty_name == "l2":
        penalty_value = l2_penalty(weights)
    else:
        raise ValueError("penalty must be 'l1' or 'l2'.")
    return mse + alpha * penalty_value

def polynomial_features(x: np.ndarray, degree: int) -> np.ndarray:
    """Create columns x^0 through x^degree without sklearn for the bias-variance page."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be one-dimensional.")
    if degree < 0:
        raise ValueError("degree cannot be negative.")
    return np.column_stack([x ** power for power in range(degree + 1)])
