# Brownian motion sampler
import numpy as np

def aggregate_brownian_increments(
        dW_fine: np.ndarray,
        factor: int,
) -> np.ndarray:
    
    if factor <= 0:
        raise ValueError("factor must be positive")
    
    if dW_fine.ndim != 2:
        raise ValueError("dW_fine needs to be a 2D array")

    n_paths, n_fine_steps = dW_fine.shape

    if n_fine_steps % factor != 0:
        raise ValueError("number of fine steps must be divisible by chosen factor")

    n_coarse_steps = n_fine_steps // factor

    return dW_fine.reshape(n_paths, n_coarse_steps, factor).sum(axis = 2)

def brownian_bridge(w_left, w_right, t_left, t_right, target_time, rng):
    # Sample W(target_time) for target_time tau in [t_left, t_right], given the
    # interval endpoints W(t_left) = w_left and W(t_right) = w_right.
    
    # The Brownian bridge conditional law is Gaussian with
    #   theta = (tau - t_left) / (t_right - t_left),
    #   mean  = w_left + theta * (w_right - w_left),
    #   var   = (tau - t_left) * (t_right - tau) / (t_right - t_left).
    
    # This couples an off-grid (adaptive) landing to the same fine Brownian
    # path as the reference. Standard NumPy broadcasting applies: e.g. w_left
    # of shape (paths,) with target_time of shape (levels, paths) gives one
    # bridge value per level and path.
    
    h = t_right - t_left

    # The clip only guards against tiny floating-point overshoot at the ends.
    tau = np.clip(target_time, t_left, t_right)

    theta = (tau - t_left) / h
    conditional_mean = w_left + theta * (w_right - w_left)
    conditional_variance = (tau - t_left) * (t_right - tau) / h

    standard_normals = rng.standard_normal(np.shape(target_time))
    return conditional_mean + np.sqrt(np.maximum(conditional_variance, 0.0)) * standard_normals