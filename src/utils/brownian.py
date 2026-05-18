# Brownian motion sampler
import numpy as np

def aggregate_brownian_increments(
        dW_fine: np.ndarray,
        factor: int,
) -> np.ndarray:
    
    if factor <0 :
        raise ValueError("factor must be positive")
    if dW_fine.ndim != 2:
        raise ValueError("dW_fine needs to be a 2D array")
    
    n_paths, n_fine_steps = dW_fine.shape

    n_coarse_steps = n_fine_steps // factor

    if n_fine_steps % factor != 0:
        raise ValueError("number of fine steps must be divisible by chosen factor")

    return dW_fine.reshape(n_paths, n_coarse_steps, factor).sum(axis = 2)