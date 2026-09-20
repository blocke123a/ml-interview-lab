"""Two-layer NumPy neural-network exercises."""
import numpy as np

def relu(z: np.ndarray) -> np.ndarray:
    """Apply ReLU elementwise."""
    z = np.asarray(z, dtype = float)
    return np.maximum(0, z)

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Apply a numerically stable sigmoid elementwise."""
    values = np.asarray(z, dtype = float)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1 / (1 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1 + exp_values)
    return result

def initialize_parameters(input_dim: int, hidden_dim: int, output_dim: int, seed: int = 42) -> dict[str, np.ndarray]:
    """Initialize correctly shaped weights and biases with suitable scaling."""
    if min(input_dim, hidden_dim, output_dim) <= 0:
        raise ValueError("Layer dimensions must be positive.")
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0, np.sqrt(2 / input_dim), (input_dim, hidden_dim)),
        "b1": np.zeros((1, hidden_dim)),
        "W2": rng.normal(0, np.sqrt(1 / hidden_dim), (hidden_dim, output_dim)),
        "b2": np.zeros((1, output_dim)),
    }

def forward_pass(X: np.ndarray, parameters: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compute hidden activations and output probabilities, returning a cache."""
    X = np.asarray(X, dtype = float)
    z1 = X @ parameters["W1"] + parameters["b1"]
    a1 = relu(z1)
    z2 = a1 @ parameters["W2"] + parameters["b2"]
    a2 = sigmoid(z2)
    return a2, {"z1": z1, "a1": a1, "z2": z2, "a2": a2}

def binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute stable average binary cross-entropy for network outputs."""
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

def backward_pass(X: np.ndarray, y: np.ndarray, parameters: dict[str, np.ndarray], cache: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Backpropagate through both layers; match gradients to parameter shapes."""
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

def update_parameters(parameters: dict[str, np.ndarray], gradients: dict[str, np.ndarray], learning_rate: float) -> dict[str, np.ndarray]:
    """Apply one gradient-descent update and return updated parameters."""
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    return {
        "W1": parameters["W1"] - learning_rate * gradients["dW1"],
        "b1": parameters["b1"] - learning_rate * gradients["db1"],
        "W2": parameters["W2"] - learning_rate * gradients["dW2"],
        "b2": parameters["b2"] - learning_rate * gradients["db2"],
    }

def train_network(X: np.ndarray, y: np.ndarray, hidden_dim: int, learning_rate: float, iterations: int, seed: int = 42) -> tuple[dict[str, np.ndarray], list[float]]:
    """Compose initialization, forward, loss, backward, and update into training."""
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
