import numpy as np
import yaml

from src.samplers.full_truncation_euler import fte_terminal
from src.utils.io import config_path
from src.utils.rng import make_rng


def load_config(filename: str) -> dict:
    with open(config_path(filename), encoding="utf-8") as f:
        return yaml.safe_load(f)


def exact_cir_mean(x0: float, kappa: float, theta: float, T: float) -> float:
    return theta + (x0 - theta) * np.exp(-kappa * T)


def test_fte_terminal_mean_close_to_exact_mean_regime_B():
    regimes_config = load_config("regimes.yaml")
    experiments_config = load_config("experiments.yaml")

    kappa = regimes_config["shared"]["kappa"]
    theta = regimes_config["shared"]["theta"]
    x0 = regimes_config["shared"]["x0"]
    sigma = regimes_config["regimes"]["B"]["sigma"]

    master_seed = experiments_config["shared"]["master_seed"]

    T = 1.0
    n_steps = 200
    n_paths = 20_000

    rng = make_rng(master_seed)

    X_T = fte_terminal(
        X0=x0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        T=T,
        n_steps=n_steps,
        n_paths=n_paths,
        rng=rng,
    )

    sample_mean = np.mean(X_T)
    expected_mean = exact_cir_mean(x0, kappa, theta, T)

    assert abs(sample_mean - expected_mean) < 0.001