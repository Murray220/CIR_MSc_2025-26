# Fixed-step strong-error comparison of FT and KL against the IF reference.
#
# Reproduces the 1/4-vs-~1 convergence-order gap from a single coupled harness:
# every scheme and the reference share one fine Brownian path, so the measured
# error is discretisation error, not variance.
#
# Run with:  uv run python experiments/run_strong_error.py

import csv

import numpy as np
import yaml

from src.metrics.strong_error import strong_errors_fixed_step, fit_loglog_order
from src.samplers.full_truncation_euler import fte_terminal_from_dW
from src.samplers.kelly_lord import kl_uniform_terminal_from_dW
from src.utils.io import config_path, results_path
from src.utils.rng import make_rng


def load_config(filename):
    with open(config_path(filename), encoding="utf-8") as f:
        return yaml.safe_load(f)


# Each scheme is a terminal-from-dW callable sharing the signature
# (X0, kappa, theta, sigma, dt, dW) -> terminal X.
SCHEMES = {
    "FT": fte_terminal_from_dW,
    "KL": kl_uniform_terminal_from_dW,
}


def local_slopes(step_sizes, errors):
    # Slope between consecutive points: log2 of the error ratio as h halves.
    # A descending sequence signals a pre-asymptotic window; a flat one means
    # the scheme is already in its asymptotic regime.
    return np.log2(errors[:-1] / errors[1:])


def run_one_scheme(name, scheme_terminal_from_dW, params, grid, rng):
    res = strong_errors_fixed_step(
        scheme_terminal_from_dW,
        X0=params["x0"],
        kappa=params["kappa"],
        theta=params["theta"],
        sigma=params["sigma"],
        T=params["T"],
        reference_n_steps=grid["reference_n_steps"],
        coarse_n_steps_list=grid["coarse_n_steps"],
        n_paths=grid["n_paths"],
        rng=rng,
    )
    dt, l1, l2 = res["dt"], res["l1"], res["l2"]
    return {
        "name": name,
        "dt": dt,
        "l1": l1,
        "l2": l2,
        "local_l2": local_slopes(dt, l2),
        "order_full": fit_loglog_order(dt, l2),
        "order_fine3": fit_loglog_order(dt[-3:], l2[-3:]),
    }


def print_report(results, params, grid):
    print()
    print(f"Strong-error comparison (kappa={params['kappa']}, "
          f"theta={params['theta']}, sigma={params['sigma']}, "
          f"X0={params['x0']}, T={params['T']})")
    print(f"Reference: IF at {grid['reference_n_steps']} steps, "
          f"{grid['n_paths']} paths")
    print("-" * 52)
    print(f"{'scheme':<8}{'order (full)':>16}{'order (fine-3)':>18}")
    for r in results:
        print(f"{r['name']:<8}{r['order_full']:>16.3f}{r['order_fine3']:>18.3f}")
    print("-" * 52)
    for r in results:
        print(f"{r['name']} local L2 slopes: {np.round(r['local_l2'], 3)}")
    print()


def save_csv(results, filename):
    path = results_path(filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scheme", "dt", "l1", "l2"])
        for r in results:
            for dt, l1, l2 in zip(r["dt"], r["l1"], r["l2"]):
                writer.writerow([r["name"], dt, l1, l2])
    return path


def main():
    regimes = load_config("regimes.yaml")
    experiments = load_config("experiments.yaml")

    shared = regimes["shared"]
    grid = experiments["time_grids"]["strong_error"]
    master_seed = experiments["shared"]["master_seed"]

    regime_name = "B"
    sigma = regimes["regimes"][regime_name]["sigma"]

    params = {
        "kappa": shared["kappa"],
        "theta": shared["theta"],
        "x0": shared["x0"],
        "sigma": sigma,
        "T": experiments["shared"]["T"],
    }

    results = []
    for name, scheme in SCHEMES.items():
        # Fresh rng per scheme, seeded identically, so FT and KL see the SAME
        # fine Brownian path and are therefore directly comparable.
        rng = make_rng(master_seed)
        results.append(run_one_scheme(name, scheme, params, grid, rng))

    print_report(results, params, grid)
    csv_path = save_csv(results, f"strong_error_regime_{regime_name}.csv")
    print(f"Saved per-step errors to {csv_path}")


if __name__ == "__main__":
    main()