import jax
import jax.numpy as jnp
import numpy as np

from klm_jax.coefficients import sigma_from_a

jax.config.update("jax_enable_x64", True)


def make_a_values(config):
    paper_values = np.linspace(
        config["a_min"],
        config["a_max"],
        config["n_paper_a_values"],
        dtype=np.float64,
    )

    if config.get("include_a0_diagnostic", False):
        return np.concatenate(([0.0], paper_values))

    return paper_values


def make_coefficients(kappa, a_values, lambda_value):
    a = jnp.asarray(a_values, dtype=jnp.float64)
    sigma = jnp.asarray(sigma_from_a(a, kappa, lambda_value), dtype=jnp.float64)

    alpha = (4.0 * kappa * lambda_value - sigma * sigma) / 8.0
    beta = -kappa / 2.0
    gamma = sigma / 2.0

    return a, sigma, alpha, beta, gamma


def implicit_lamperti_step(y_old, brownian_increment, step_size, alpha, beta, gamma):
    u = y_old + gamma * brownian_increment
    denominator = 1.0 - beta * step_size

    return (
        u / (2.0 * denominator)
        + jnp.sqrt(
            u * u / (4.0 * denominator * denominator)
            + alpha * step_size / denominator
        )
    )


def choose_adaptive_step_size(y_values, current_times, hmax_by_level, rho, final_time):
    proposed = hmax_by_level[None, :, None] * jnp.minimum(1.0, jnp.abs(y_values))
    minimum = (hmax_by_level / rho)[None, :, None]

    used_minimum_step = proposed <= minimum
    chosen = jnp.where(used_minimum_step, minimum, proposed)

    remaining_time = final_time - current_times
    return jnp.minimum(chosen, remaining_time), used_minimum_step


def root_mean_square_error(scheme_x_values, reference_x_values):
    errors = scheme_x_values - reference_x_values[:, None, :]
    return jnp.sqrt(jnp.mean(errors * errors, axis=2))


def fit_orders(step_sizes, errors):
    log_h = jnp.log(step_sizes)
    log_e = jnp.log(errors)

    centered_h = log_h - jnp.mean(log_h, axis=1, keepdims=True)
    centered_e = log_e - jnp.mean(log_e, axis=1, keepdims=True)

    numerator = jnp.sum(centered_h * centered_e, axis=1)
    denominator = jnp.sum(centered_h * centered_h, axis=1)

    return numerator / denominator


def run_fig3_experiment(config, outdir=None):
    lambda_value = config["lambda_value"]
    reference_power = config["reference_power"]
    reference_step = 2.0 ** (-reference_power)
    number_of_fine_steps = 2**reference_power

    a_values = make_a_values(config)
    hmax_by_level = np.array([2.0 ** (-level) for level in config["levels"]])

    rows = []

    for kappa in config["kappas"]:
        a, sigma, alpha, beta, gamma = make_coefficients(
            kappa,
            a_values,
            lambda_value,
        )

        for a_i, sigma_i, alpha_i, gamma_i in zip(a, sigma, alpha, gamma, strict=True):
            for level, hmax in zip(config["levels"], hmax_by_level, strict=True):
                rows.append(
                    {
                        "kappa": float(kappa),
                        "a": float(a_i),
                        "sigma": float(sigma_i),
                        "alpha": float(alpha_i),
                        "beta": float(beta),
                        "gamma": float(gamma_i),
                        "level": int(level),
                        "hmax": float(hmax),
                        "reference_power": int(reference_power),
                        "reference_step": float(reference_step),
                        "number_of_fine_steps": int(number_of_fine_steps),
                        "n_paths": int(config["n_paths"]),
                    }
                )

    return rows