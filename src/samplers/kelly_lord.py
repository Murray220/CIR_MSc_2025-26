# Uniform mesh, alpha >=0, i.e this is in regimes A,B,C only.
import numpy as np

from src.utils.cir_params import kl_alpha

def kl_uniform_step(
        x: np.ndarray,
        kappa: float,
        theta: float,
        sigma: float,
        dt: float,
        dW: np.ndarray,
)-> np.ndarray:
    
    alpha = kl_alpha(kappa, theta, sigma)

    # Check alpha 
    if alpha < 0.0:
        raise ValueError(
            "Error alpha <0"
        )
    

    inside_sqrt = x + 2.0 * alpha * dt

    return np.exp(-kappa * dt) * (np.sqrt(inside_sqrt) + 0.5 * sigma * dW) ** 2

def kl_uniform_paths_from_dW(
        X0: float,
        kappa: float,
        theta: float,
        sigma: float,
        dt: float,
        dW: np.ndarray,
) -> np.ndarray:
    
    alpha = kl_alpha(kappa, theta, sigma)

    # Check alpha 
    if alpha < 0.0:
        raise ValueError(
            "Error alpha <0"
        )
    
    n_paths, n_steps = dW.shape
    
    X = np.empty((n_paths, n_steps + 1), dtype = float)
    X[:, 0] = X0

    for n in range(n_steps):
        X[:, n+1] = kl_uniform_step(
            x = X[:, n],
            kappa = kappa,
            theta = theta,
            sigma = sigma,
            dt = dt,
            dW = dW[:, n],
        )

    return X

def kl_uniform_terminal_from_dW(
        X0: float,
        kappa: float,
        theta: float,
        sigma: float,
        dt: float,
        dW: np.ndarray,
) -> np.ndarray:
    
    alpha = kl_alpha(kappa, theta, sigma)

    # Check alpha 
    if alpha < 0.0:
        raise ValueError(
            "Error alpha <0"
        )
    
    n_paths, n_steps = dW.shape

    x = np.full(n_paths, X0, dtype = float)

    for n in range(n_steps):
        x = kl_uniform_step(
            x = x,
            kappa = kappa,
            theta = theta,
            sigma = sigma,
            dt = dt,
            dW = dW[:,n]
        )

    return x

def kl_uniform_paths(
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

    return kl_uniform_paths_from_dW(
        X0 = X0,
        kappa = kappa,
        theta = theta,
        sigma = sigma,
        dt = dt,
        dW = dW,
    )

def kl_uniform_terminal(
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

    return kl_uniform_terminal_from_dW(
        X0 = X0,
        kappa = kappa,
        theta = theta,
        sigma = sigma,
        dt = dt,
        dW = dW,
    )