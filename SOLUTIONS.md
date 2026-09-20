# ML Interview Lab — Solution Reference

Use this as a checking guide after attempting each exercise. The exercise modules should remain your working versions.

## Gradient descent

```python
from collections.abc import Callable


def objective_function(x: float) -> float:
    return (x + 1) ** 2


def objective_gradient(x: float) -> float:
    return 2 * x + 2


def gradient_descent(
    gradient_fn: Callable[[float], float],
    initial_x: float,
    learning_rate: float,
    iterations: int,
) -> tuple[list[float], list[float]]:
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if iterations < 0:
        raise ValueError("iterations cannot be negative.")

    current_x = float(initial_x)
    x_history = [current_x]
    loss_history = [objective_function(current_x)]

    for _ in range(iterations):
        current_x -= learning_rate * gradient_fn(current_x)
        x_history.append(current_x)
        loss_history.append(objective_function(current_x))

    return x_history, loss_history
```

## Linear regression

```python
import numpy as np


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("Inputs must have the same nonzero shape.")
    return float(np.mean((y_true - y_pred) ** 2))


def fit_simple_ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be one-dimensional.")
    if x.size == 0 or x.size != y.size:
        raise ValueError("x and y must have the same nonzero length.")

    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denominator = np.sum(x_centered ** 2)
    if np.isclose(denominator, 0):
        raise ValueError("Cannot estimate a slope when x has no variation.")

    slope = np.sum(x_centered * y_centered) / denominator
    intercept = np.mean(y) - slope * np.mean(x)
    return float(intercept), float(slope)


def predict(x: np.ndarray, intercept: float, slope: float) -> np.ndarray:
    return intercept + slope * np.asarray(x, dtype=float)


def residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("Inputs must have the same shape.")
    return y_true - y_pred


def matrix_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D.")
    if X.shape[0] == 0 or X.shape[0] != y.size:
        raise ValueError("X and y must have the same nonzero sample count.")

    design = np.column_stack((np.ones(X.shape[0]), X))
    return np.linalg.pinv(design) @ y
```

## Logistic regression

```python
import numpy as np


def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    values = np.asarray(z, dtype=float)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1 / (1 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1 + exp_values)
    return float(result) if np.isscalar(z) else result


def binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape or y_true.size == 0:
        raise ValueError("Inputs must have the same nonzero shape.")
    if np.any(~np.isin(y_true, [0, 1])):
        raise ValueError("Labels must be binary.")
    epsilon = np.finfo(float).eps
    probabilities = np.clip(y_prob, epsilon, 1 - epsilon)
    return float(-np.mean(
        y_true * np.log(probabilities)
        + (1 - y_true) * np.log(1 - probabilities)
    ))


def linear_score(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    return np.asarray(X, dtype=float) @ np.asarray(weights, dtype=float) + bias


def compute_gradients(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bias: float,
) -> tuple[np.ndarray, float]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    probabilities = sigmoid(linear_score(X, weights, bias))
    errors = probabilities - y
    weight_gradient = X.T @ errors / X.shape[0]
    bias_gradient = float(np.mean(errors))
    return weight_gradient, bias_gradient


def fit_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float,
    iterations: int,
) -> tuple[np.ndarray, float, list[float]]:
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
    return np.asarray(sigmoid(linear_score(X, weights, bias)))


def predict_class(
    X: np.ndarray,
    weights: np.ndarray,
    bias: float,
    threshold: float = 0.5,
) -> np.ndarray:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")
    return (predict_proba(X, weights, bias) >= threshold).astype(int)
```

## Regularization and polynomial features

This convention assumes `weights` excludes the intercept and defines regularized loss as `MSE + alpha * penalty`.

```python
import numpy as np


def l1_penalty(weights: np.ndarray) -> float:
    return float(np.sum(np.abs(np.asarray(weights, dtype=float))))


def l2_penalty(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    return float(np.sum(weights ** 2))


def regularized_mse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    penalty: str,
) -> float:
    if alpha < 0:
        raise ValueError("alpha cannot be negative.")
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("Targets and predictions must have the same nonzero shape.")

    mse = float(np.mean((y_true - y_pred) ** 2))
    penalty_name = penalty.lower()
    if penalty_name == "l1":
        penalty_value = l1_penalty(weights)
    elif penalty_name == "l2":
        penalty_value = l2_penalty(weights)
    else:
        raise ValueError("penalty must be 'l1' or 'l2'.")
    return mse + alpha * penalty_value


def polynomial_features(x: np.ndarray, degree: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be one-dimensional.")
    if degree < 0:
        raise ValueError("degree cannot be negative.")
    return np.column_stack([x ** power for power in range(degree + 1)])
```

## Bayesian updating

```python
from scipy.stats import beta as beta_distribution


def beta_binomial_update(
    prior_alpha: float,
    prior_beta: float,
    successes: int,
    failures: int,
) -> tuple[float, float]:
    if prior_alpha <= 0 or prior_beta <= 0:
        raise ValueError("Prior parameters must be positive.")
    if successes < 0 or failures < 0:
        raise ValueError("Observation counts cannot be negative.")
    return prior_alpha + successes, prior_beta + failures


def posterior_mean(alpha: float, beta: float) -> float:
    if alpha <= 0 or beta <= 0:
        raise ValueError("Beta parameters must be positive.")
    return alpha / (alpha + beta)


def credible_interval(
    alpha: float,
    beta: float,
    level: float = 0.95,
) -> tuple[float, float]:
    if alpha <= 0 or beta <= 0:
        raise ValueError("Beta parameters must be positive.")
    if not 0 < level < 1:
        raise ValueError("level must be strictly between 0 and 1.")
    tail = (1 - level) / 2
    lower, upper = beta_distribution.ppf([tail, 1 - tail], alpha, beta)
    return float(lower), float(upper)
```

## Neural network

This solution uses a ReLU hidden layer, sigmoid output, and binary cross-entropy. Shapes are `X: (n, d)`, `y: (n,)`, and predictions: `(n, 1)`.

```python
import numpy as np


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, np.asarray(z, dtype=float))


def sigmoid(z: np.ndarray) -> np.ndarray:
    values = np.asarray(z, dtype=float)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1 / (1 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1 + exp_values)
    return result


def initialize_parameters(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    if min(input_dim, hidden_dim, output_dim) <= 0:
        raise ValueError("Layer dimensions must be positive.")
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0, np.sqrt(2 / input_dim), (input_dim, hidden_dim)),
        "b1": np.zeros((1, hidden_dim)),
        "W2": rng.normal(0, np.sqrt(1 / hidden_dim), (hidden_dim, output_dim)),
        "b2": np.zeros((1, output_dim)),
    }


def forward_pass(
    X: np.ndarray,
    parameters: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    X = np.asarray(X, dtype=float)
    z1 = X @ parameters["W1"] + parameters["b1"]
    a1 = relu(z1)
    z2 = a1 @ parameters["W2"] + parameters["b2"]
    a2 = sigmoid(z2)
    return a2, {"z1": z1, "a1": a1, "z2": z2, "a2": a2}


def binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1, 1)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape or y_true.size == 0:
        raise ValueError("Targets and probabilities must have matching nonzero shapes.")
    epsilon = np.finfo(float).eps
    probabilities = np.clip(y_prob, epsilon, 1 - epsilon)
    return float(-np.mean(
        y_true * np.log(probabilities)
        + (1 - y_true) * np.log(1 - probabilities)
    ))


def backward_pass(
    X: np.ndarray,
    y: np.ndarray,
    parameters: dict[str, np.ndarray],
    cache: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    X = np.asarray(X, dtype=float)
    targets = np.asarray(y, dtype=float).reshape(-1, 1)
    sample_count = X.shape[0]

    dz2 = (cache["a2"] - targets) / sample_count
    dW2 = cache["a1"].T @ dz2
    db2 = np.sum(dz2, axis=0, keepdims=True)
    da1 = dz2 @ parameters["W2"].T
    dz1 = da1 * (cache["z1"] > 0)
    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)
    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}


def update_parameters(
    parameters: dict[str, np.ndarray],
    gradients: dict[str, np.ndarray],
    learning_rate: float,
) -> dict[str, np.ndarray]:
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    return {
        "W1": parameters["W1"] - learning_rate * gradients["dW1"],
        "b1": parameters["b1"] - learning_rate * gradients["db1"],
        "W2": parameters["W2"] - learning_rate * gradients["dW2"],
        "b2": parameters["b2"] - learning_rate * gradients["db2"],
    }


def train_network(
    X: np.ndarray,
    y: np.ndarray,
    hidden_dim: int,
    learning_rate: float,
    iterations: int,
    seed: int = 42,
) -> tuple[dict[str, np.ndarray], list[float]]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.size:
        raise ValueError("X must be 2D and align with one-dimensional y.")
    if iterations < 0:
        raise ValueError("iterations cannot be negative.")

    parameters = initialize_parameters(X.shape[1], hidden_dim, 1, seed)
    loss_history = []
    for _ in range(iterations):
        probabilities, cache = forward_pass(X, parameters)
        loss_history.append(binary_cross_entropy(y, probabilities))
        gradients = backward_pass(X, y, parameters, cache)
        parameters = update_parameters(parameters, gradients, learning_rate)
    return parameters, loss_history
```

## Model evaluation

Undefined-metric convention: raise `ValueError`, except the empty-prediction endpoint of a precision-recall curve, where precision is conventionally set to `1.0`.

```python
import numpy as np


def _validate_binary_inputs(y_true: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true)
    values = np.asarray(values)
    if y_true.ndim != 1 or values.ndim != 1:
        raise ValueError("Inputs must be one-dimensional.")
    if y_true.size == 0 or y_true.size != values.size:
        raise ValueError("Inputs must have the same nonzero length.")
    if np.any(~np.isin(y_true, [0, 1])):
        raise ValueError("Labels must be binary.")
    return y_true, values


def confusion_matrix_from_scratch(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[int, int, int, int]:
    y_true, y_pred = _validate_binary_inputs(y_true, y_pred)
    if np.any(~np.isin(y_pred, [0, 1])):
        raise ValueError("Predictions must be binary.")
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return tn, fp, fn, tp


def accuracy(tn: int, fp: int, fn: int, tp: int) -> float:
    total = tn + fp + fn + tp
    if total == 0:
        raise ValueError("Accuracy is undefined without observations.")
    return (tn + tp) / total


def precision(tn: int, fp: int, fn: int, tp: int) -> float:
    denominator = tp + fp
    if denominator == 0:
        raise ValueError("Precision is undefined without predicted positives.")
    return tp / denominator


def recall(tn: int, fp: int, fn: int, tp: int) -> float:
    denominator = tp + fn
    if denominator == 0:
        raise ValueError("Recall is undefined without actual positives.")
    return tp / denominator


def specificity(tn: int, fp: int, fn: int, tp: int) -> float:
    denominator = tn + fp
    if denominator == 0:
        raise ValueError("Specificity is undefined without actual negatives.")
    return tn / denominator


def f1_score(tn: int, fp: int, fn: int, tp: int) -> float:
    denominator = 2 * tp + fp + fn
    if denominator == 0:
        raise ValueError("F1 is undefined without actual or predicted positives.")
    return 2 * tp / denominator


def _validate_probabilities(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_true, y_prob = _validate_binary_inputs(y_true, y_prob)
    y_prob = y_prob.astype(float)
    if np.any(~np.isfinite(y_prob)) or np.any((y_prob < 0) | (y_prob > 1)):
        raise ValueError("Probabilities must be finite and between 0 and 1.")
    return y_true, y_prob


def roc_points(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true, y_prob = _validate_probabilities(y_true, y_prob)
    positive_count = int(np.sum(y_true == 1))
    negative_count = int(np.sum(y_true == 0))
    if positive_count == 0 or negative_count == 0:
        raise ValueError("ROC requires both positive and negative observations.")

    thresholds = np.concatenate(([np.inf], np.sort(np.unique(y_prob))[::-1]))
    fpr_values = []
    tpr_values = []
    for threshold in thresholds:
        predictions = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix_from_scratch(y_true, predictions)
        fpr_values.append(fp / (fp + tn))
        tpr_values.append(tp / (tp + fn))
    return np.asarray(fpr_values), np.asarray(tpr_values), thresholds


def precision_recall_points(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
```

## Suggested checks

- Compare OLS coefficients and predictions with scikit-learn.
- Compare classification metrics and curve points with scikit-learn, accounting for endpoint conventions.
- Numerically check logistic and neural-network gradients with finite differences.
- Confirm loss falls on a small, deterministic training problem.
- Test empty inputs, incompatible shapes, constant predictors, invalid labels, extreme logits, and zero metric denominators.

## Clustering

~~~python
import numpy as np

def euclidean_distance(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("Vectors must have equal shapes.")
    return float(np.sqrt(np.sum((a - b) ** 2)))

def initialize_centroids(X, n_clusters, seed=42):
    X = np.asarray(X, float)
    if X.ndim != 2 or not 1 <= n_clusters <= len(X):
        raise ValueError("Use a 2D array and a valid cluster count.")
    rng = np.random.default_rng(seed)
    return X[rng.choice(len(X), n_clusters, replace=False)].copy()

def assign_clusters(X, centroids):
    X, centroids = np.asarray(X, float), np.asarray(centroids, float)
    distances = np.sqrt(np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2))
    return np.argmin(distances, axis=1)

def update_centroids(X, labels, n_clusters):
    X, labels = np.asarray(X, float), np.asarray(labels)
    centroids = []
    for cluster in range(n_clusters):
        members = X[labels == cluster]
        centroids.append(members.mean(axis=0) if len(members) else X[cluster % len(X)])
    return np.asarray(centroids)

def fit_kmeans(X, n_clusters, max_iterations=100, tolerance=1e-4, seed=42):
    X = np.asarray(X, float)
    centroids = initialize_centroids(X, n_clusters, seed)
    inertia_history = []
    for _ in range(max_iterations):
        labels = assign_clusters(X, centroids)
        inertia_history.append(float(np.sum((X - centroids[labels]) ** 2)))
        new_centroids = update_centroids(X, labels, n_clusters)
        if np.linalg.norm(new_centroids - centroids) <= tolerance:
            centroids = new_centroids
            labels = assign_clusters(X, centroids)
            break
        centroids = new_centroids
    return centroids, labels, inertia_history

def region_query(X, point_index, epsilon):
    X = np.asarray(X, float)
    distances = np.sqrt(np.sum((X - X[point_index]) ** 2, axis=1))
    return np.flatnonzero(distances <= epsilon)

def dbscan(X, epsilon, min_samples):
    X = np.asarray(X, float)
    if epsilon <= 0 or min_samples <= 0:
        raise ValueError("epsilon and min_samples must be positive.")
    unvisited, noise = -2, -1
    labels = np.full(len(X), unvisited, dtype=int)
    cluster = 0
    for point in range(len(X)):
        if labels[point] != unvisited:
            continue
        neighbors = region_query(X, point, epsilon)
        if len(neighbors) < min_samples:
            labels[point] = noise
            continue
        labels[point] = cluster
        seeds = list(neighbors[neighbors != point])
        while seeds:
            neighbor = seeds.pop()
            if labels[neighbor] == noise:
                labels[neighbor] = cluster
            if labels[neighbor] != unvisited:
                continue
            labels[neighbor] = cluster
            nearby = region_query(X, neighbor, epsilon)
            if len(nearby) >= min_samples:
                seeds.extend(i for i in nearby if labels[i] in (unvisited, noise))
        cluster += 1
    return labels

def silhouette_score_from_scratch(X, labels):
    X, labels = np.asarray(X, float), np.asarray(labels)
    valid = labels != -1
    X, labels = X[valid], labels[valid]
    clusters = np.unique(labels)
    if len(clusters) < 2:
        raise ValueError("At least two non-noise clusters are required.")
    distances = np.sqrt(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2))
    scores = []
    for i, cluster in enumerate(labels):
        same = np.flatnonzero(labels == cluster)
        same = same[same != i]
        a = distances[i, same].mean() if len(same) else 0.0
        b = min(distances[i, labels == other].mean() for other in clusters if other != cluster)
        scores.append(0.0 if max(a, b) == 0 else (b - a) / max(a, b))
    return float(np.mean(scores))
~~~

## Forecasting

~~~python
import numpy as np

def naive_forecast(series, horizon=1):
    series = np.asarray(series, float)
    if series.ndim != 1 or len(series) == 0 or horizon <= 0:
        raise ValueError("Use a nonempty 1D series and positive horizon.")
    return np.repeat(series[-1], horizon)

def seasonal_naive_forecast(series, season_length, horizon):
    series = np.asarray(series, float)
    if series.ndim != 1 or season_length <= 0 or horizon <= 0 or len(series) < season_length:
        raise ValueError("Invalid series, season length, or horizon.")
    return np.resize(series[-season_length:], horizon)

def moving_average_forecast(series, window, horizon=1):
    history = list(np.asarray(series, float))
    if window <= 0 or horizon <= 0 or len(history) < window:
        raise ValueError("Invalid window or horizon.")
    forecasts = []
    for _ in range(horizon):
        value = float(np.mean(history[-window:]))
        forecasts.append(value)
        history.append(value)
    return np.asarray(forecasts)

def simple_exponential_smoothing(series, alpha):
    series = np.asarray(series, float)
    if series.ndim != 1 or len(series) == 0 or not 0 < alpha <= 1:
        raise ValueError("Use a nonempty series and alpha in (0, 1].")
    levels = np.empty_like(series)
    levels[0] = series[0]
    for index in range(1, len(series)):
        levels[index] = alpha * series[index] + (1 - alpha) * levels[index - 1]
    return levels, float(levels[-1])

def mean_absolute_error(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("Inputs must have equal nonzero shapes.")
    return float(np.mean(np.abs(y_true - y_pred)))

def rolling_origin_splits(n_observations, initial_window, horizon, step=1):
    if min(n_observations, initial_window, horizon, step) <= 0 or initial_window + horizon > n_observations:
        raise ValueError("Invalid rolling-origin configuration.")
    splits = []
    train_end = initial_window
    while train_end + horizon <= n_observations:
        splits.append((np.arange(train_end), np.arange(train_end, train_end + horizon)))
        train_end += step
    return splits
~~~
