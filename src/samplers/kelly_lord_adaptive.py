# for regimes D-E
import numpy as np

from src.utils.cir_params import kl_alpha


def soft_zero_threshold(kappa, theta, dt_max, rho=2.0):
    # Soft-zero boundary X_zero 
    return theta * (1.0 - np.exp(-kappa * dt_max)) / rho


def kl_adaptive_terminal(
        X0: float,
        kappa: float,
        theta: float,
        sigma: float,
        T: float,
        dt_max: float,
        n_paths: int,
        rng: np.random.Generator,
        rho: float = 2.0,
        max_rounds: int = 1_000_000,
) -> np.ndarray:
    alpha = kl_alpha(kappa, theta, sigma)
    X_zero = soft_zero_threshold(kappa, theta, dt_max, rho)

    x = np.full(n_paths, X0, dtype=float)
    t = np.zeros(n_paths, dtype=float)

    rounds = 0
    # Each path advances until it reaches T. Finished paths are masked out.
    while np.any(t < T - 1e-12):
        rounds += 1
        if rounds > max_rounds:
            raise RuntimeError("kl_adaptive did not reach T; check parameters")

        active = t < T - 1e-12
        dt_remaining = T - t

        in_soft_zero = active & (x < X_zero)
        in_splitting = active & ~in_soft_zero

        dt = np.zeros(n_paths, dtype=float)

        # --- soft-zero region deterministic ODE
        if np.any(in_soft_zero):
            xs = x[in_soft_zero]
            dt_sz = -np.log((X_zero - theta) / (xs - theta)) / kappa
            dt[in_soft_zero] = np.minimum(dt_sz, dt_remaining[in_soft_zero])

        # --- splitting region do the adaptive step
        if np.any(in_splitting):
            if alpha < 0.0:
                # Shrink so the inner sqrt argument stays positive
                dt_adaptive = 0.95 * x[in_splitting] / (2.0 * abs(alpha))
                dt[in_splitting] = np.minimum(
                    np.minimum(dt_adaptive, dt_max), dt_remaining[in_splitting]
                )
            else:
                # alpha >= 0: positivity is automatic; no shrinking needed.
                dt[in_splitting] = np.minimum(dt_max, dt_remaining[in_splitting])

        x_next = x.copy()

        # apply soft-zero deterministic flow
        if np.any(in_soft_zero):
            h = dt[in_soft_zero]
            decay = np.exp(-kappa * h)
            x_next[in_soft_zero] = decay * x[in_soft_zero] + theta * (1.0 - decay)

        # apply splitting update
        if np.any(in_splitting):
            h = dt[in_splitting]
            dW = np.sqrt(h) * rng.standard_normal(np.count_nonzero(in_splitting))
            inside_sqrt = x[in_splitting] + 2.0 * alpha * h
            # stop floating point round off such that it guarantees inside_sqrt > 0.
            if np.any(inside_sqrt < -1e-14):
                raise RuntimeError("negative square-root argument")
            inside_sqrt = np.maximum(inside_sqrt, 0.0)

            x_next[in_splitting] = np.exp(-kappa * h) * (
                np.sqrt(inside_sqrt) + 0.5 * sigma * dW
            ) ** 2

        x = x_next
        t = t + dt

    return x