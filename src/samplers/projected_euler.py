import numpy as np

from src.utils.cir_params import kl_alpha
from src.utils.rng import make_brownian_increments


def projected_euler_step(
    y: np.ndarray,
    alpha: float,
    kappa: float,
    sigma: float,
    dt: float,
    dW: np.ndarray,
    y_floor: float,
) -> np.ndarray:
    if y_floor <= 0.0:
        raise ValueError("y_floor must be positive")

    y_safe = np.maximum(y, y_floor)

    y_hat = (
        y_safe
        + (alpha / y_safe - 0.5 * kappa * y_safe) * dt
        + 0.5 * sigma * dW
    )

    return np.maximum(y_hat, y_floor)


def projected_euler_paths_from_dW(
    X0: float,
    kappa: float,
    theta: float,
    sigma: float,
    dt: float,
    dW: np.ndarray,
    y_floor: float | None = None,
) -> np.ndarray:
    alpha = kl_alpha(kappa, theta, sigma)

    if y_floor is None:
        y_floor = dt

    n_paths, n_steps = dW.shape

    Y = np.empty((n_paths, n_steps + 1), dtype=float)
    Y[:, 0] = max(np.sqrt(X0), y_floor)

    for n in range(n_steps):
        Y[:, n + 1] = projected_euler_step(
            y=Y[:, n],
            alpha=alpha,
            kappa=kappa,
            sigma=sigma,
            dt=dt,
            dW=dW[:, n],
            y_floor=y_floor,
        )

    return Y**2


def projected_euler_terminal_from_dW(
    X0: float,
    kappa: float,
    theta: float,
    sigma: float,
    dt: float,
    dW: np.ndarray,
    y_floor: float | None = None,
) -> np.ndarray:
    alpha = kl_alpha(kappa, theta, sigma)

    if y_floor is None:
        y_floor = dt

    n_paths, n_steps = dW.shape
    y = np.full(n_paths, max(np.sqrt(X0), y_floor), dtype=float)

    for n in range(n_steps):
        y = projected_euler_step(
            y=y,
            alpha=alpha,
            kappa=kappa,
            sigma=sigma,
            dt=dt,
            dW=dW[:, n],
            y_floor=y_floor,
        )

    return y**2


def projected_euler_paths(
    X0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
    y_floor: float | None = None,
) -> np.ndarray:
    dt = T / n_steps
    dW = make_brownian_increments(rng, n_paths, n_steps, dt)

    return projected_euler_paths_from_dW(
        X0=X0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        dt=dt,
        dW=dW,
        y_floor=y_floor,
    )


def projected_euler_terminal(
    X0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
    y_floor: float | None = None,
) -> np.ndarray:
    dt = T / n_steps
    dW = make_brownian_increments(rng, n_paths, n_steps, dt)

    return projected_euler_terminal_from_dW(
        X0=X0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        dt=dt,
        dW=dW,
        y_floor=y_floor,
    )