
from __future__ import annotations
import numpy as np

# Full truncation euler 

# Simulate a single step/update for the full truncation euler scheme
def fte_step(
    x: np.ndarray,
    kappa: float,
    theta: float,
    sigma: float,
    dt: float,
    dW: np.ndarray
) -> np.darray:

    x_pos = np.maximum(x, 0.0)

    x_hat = (
        x + kappa * (theta - x_pos) * dt + sigma * np.sqrt(x_pos) * dW
    )

    return np.maximum(x_hat, 0.0)


# Simulate FULL paths generated from dW (using pre-generated Brownian Incremenets) in the FTE scheme
def fte_paths_from_dW(
        X0: float,
        kappa: float,
        theta: float,
        sigma: float,
        dt: float,
        dW: np.ndarray,
) -> np.darray:
    
    n_paths, n_steps = dW.shape

    X = np.empty( (n_paths, n_steps + 1),
                 dtype = float)
    X[:, 0] = X0

    for n in range(n_steps):
        X[:, n +1] = fte_step(
            x = X[:, n],
            kappa = kappa,
            theta = theta,
            sigma = sigma,
            dt = dt,
            dW = dW[:,n],
        )
    return X

# Simulate terminal FTE values only.
def fte_terminal_from_dW(
        X0 : float,
        kappa: float,
        theta: float,
        sigma: float,
        dt: float,
        dW: np.ndarray,
) -> np.ndarray:

    n_paths, n_steps = dW.shape

    x = np.full(n_paths, X0, dtype = float)

    for n in range(n_steps):
        x = fte_step(
            x = x,
            kappa = kappa,
            theta = theta,
            sigma = sigma,
            dt = dt,
            dW = dW[:, n]
        )

    return x

def fte_paths(
        X0: float,
        kappa: float,
        theta: float,
        sigma: float,
        T: float,
        n_steps: int,
        n_paths: int,
        rng: np.random.Generator,
) -> np.ndarray:
    
    dt = T / n_steps
    dW = np.sqrt(dt) * rng.standard_normal((n_paths, n_steps))

    return fte_paths_from_dW(
        X0 = X0,
        kappa = kappa,
        theta = theta,
        sigma = sigma,
        dt = dt,
        dW = dW,
    )

def fte_terminal(
        X0: float,
        kappa: float,
        theta: float,
        sigma: float,
        T: float,
        n_steps: int,
        n_paths: int,
        rng: np.random.Generator,
) -> np.ndarray:
    
    dt = T / n_steps
    dW = np.sqrt(dt) * rng.standard_normal((n_paths, n_steps))

    return fte_terminal_from_dW(
        X0=X0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        dt=dt,
        dW=dW,
    )