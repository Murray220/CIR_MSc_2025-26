import numpy as np
import yaml

from src.samplers.full_truncation_euler import fte_paths
from src.utils.io import config_path
from src.utils.rng import make_rng

def load_config(filename: str) -> dict:
    with open(config_path(filename), encoding="utf-8") as f:
        return yaml.safe_load(f)
    
def test_fte_paths_nonnegative_all_regimes():
    regimes_config = load_config("regimes.yaml")
    experiments_config = load_config("experiments.yaml")

    kappa = regimes_config["shared"]["kappa"]
    theta = regimes_config["shared"]["theta"]
    x0 = regimes_config["shared"]["x0"]

    master_seed = experiments_config["shared"]["master_seed"]

    experiment = experiments_config["experiments"]["fte_all_regime_smoke"]

    T = experiment["T"]
    n_steps = experiment["n_steps"]
    n_paths = experiment["n_paths"]

    for i, regime_name in enumerate(experiment["regimes"]):
        sigma = regimes_config["regimes"][regime_name]["sigma"]

        rng = make_rng(master_seed + 1)

        X = fte_paths(
            X0 = x0,
            kappa = kappa,
            theta = theta,
            sigma = sigma,
            T = T,
            n_steps = n_steps,
            n_paths = n_paths,
            rng = rng,
        )

        assert np.all(X >= 0.0), f"FTE produced negative values in regime {regime_name}"
