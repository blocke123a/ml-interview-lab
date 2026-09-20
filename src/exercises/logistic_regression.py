"""Binary logistic-regression exercises."""
import numpy as np


def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    """Map scores to probabilities in (0, 1).

    Implement a numerically stable version that behaves well for large magnitudes.
    """
    values = np.asarray(z, dtype=float)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1 / (1 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1 + exp_values)
    return float(result) if np.isscalar(z) else result

def binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Return average binary cross-entropy.

    Guard logarithms against exact zero and one without changing the labels.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape or y_true.size == 0:
        raise ValueError("Inputs must have the same nonzero shape.")
    if (np.any(~np.isin(y_true, [0, 1]))):
        raise ValueError("Labels must be binary.")
    epsilon = np.finfo(float).eps
    probabilities = np.clip(y_prob, epsilon, 1 - epsilon)
    return float(-np.mean(y_true * np.log(probabilities) + (1 - y_true) * np.log(1 - probabilities)))

def linear_score(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    """Compute the affine score (logit) for every row of X."""
    X = np.asarray(X, dtype=float)
    weights = np.asarray(weights, dtype=float)
    return X @ weights + bias


def compute_gradients(X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float) -> tuple[np.ndarray, float]:
    """Return BCE gradients for weights and bias.

    Derive these from the chain rule and average over the observations.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    weights = np.asarray(weights, dtype=float)
    probabilities = sigmoid(linear_score(X, weights, bias))
    errors = probabilities - y
    weight_gradient = X.T @ errors / X.shape[0]
    bias_gradient = float(np.mean(errors))
    return weight_gradient, bias_gradient

def fit_logistic_regression(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[np.ndarray, float, list[float]]:
    """Train binary logistic regression with batch gradient descent.

    Initialize parameters, update them with ``compute_gradients``, and retain the
    loss at each iteration so convergence can be inspected.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.size:
        raise ValueError("X must be 2D and align with one-dimensional y.")
    if learning_rate <= 0 or iterations < 0:
        raise ValueError("Use a positive learning rate and nonnegative iterations.")
    weights = np.zeros(X.shape[1], dtype=float)
    bias = 0.0
    loss_history = []
    for _ in range(iterations):
        weight_gradient, bias_gradient = compute_gradients(X, y, weights, bias)
        weights -= learning_rate * weight_gradient
        bias -= learning_rate * bias_gradient
        loss_history.append(binary_cross_entropy(y, predict_proba(X, weights, bias)))
    return weights, bias, loss_history

def predict_proba(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    """Return positive-class probabilities for each row."""
    return np.asarray(sigmoid(linear_score(X, weights, bias)))

def predict_class(X: np.ndarray, weights: np.ndarray, bias: float, threshold: float = 0.5) -> np.ndarray:
    """Convert positive-class probabilities to integer labels using ``threshold``."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")
    return (predict_proba(X, weights, bias) >= threshold).astype(int)

