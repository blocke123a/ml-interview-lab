"""Optimization exercises. Implement these without an optimizer library."""
from collections.abc import Callable


def objective_function(x: float) -> float:
    """Return f(x) = (x + 1)^2.

    Implement the scalar objective used by the playground. Hint: expand the square
    only if it helps you differentiate it.
    """
    return (x + 1) ** 2


def objective_gradient(x: float) -> float:
    """Return the analytical derivative of :func:`objective_function` at ``x``.

    Differentiate the objective by hand; do not use automatic differentiation.
    """
    return (2 * x) + 2


def gradient_descent(
    gradient_fn: Callable[[float], float], initial_x: float, learning_rate: float, iterations: int
) -> tuple[list[float], list[float]]:
    """Iteratively minimize a scalar function and return parameter and loss histories.

    Apply the gradient-descent update for exactly ``iterations`` steps. Include the
    initial point in both histories and evaluate losses with ``objective_function``.
    Hint: each new value depends on the gradient at the current value.
    """
    x_history = [initial_x]
    loss_history = [objective_function(initial_x)]
    current_x = initial_x
    for _ in range(iterations):
        current_x -= learning_rate * gradient_fn(current_x)
        x_history.append(current_x)
        loss_history.append(objective_function(current_x))

    return x_history, loss_history
