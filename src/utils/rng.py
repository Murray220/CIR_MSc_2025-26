# random number generator script

import numpy as np

def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)

def make_brownian_increments(
        rng: np.random.Generator,
        n_paths: int,
        n_steps: int,
        dt: float,
) -> np.ndarray:
    return np.sqrt(dt) * rng.standard_normal((n_paths, n_steps))
