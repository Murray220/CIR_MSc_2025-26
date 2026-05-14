import yaml

from src.utils.io import config_path


# Takes a YAML filename and returns its contents as a Python dictionary.
def _load_yaml(filename: str) -> dict:
    with open(config_path(filename), encoding="utf-8") as f:
        return yaml.safe_load(f)


# Checks that the current (as of 14.05.2026) experiment names exist in experiments.yaml.
def test_experiments_config_has_expected_experiments():
    config = _load_yaml("experiments.yaml")

    expected = {
        "exact_transition_figure",
        "fte_vs_exact_sanity",
        "fte_all_regime_smoke",
    }

    assert expected.issubset(config["experiments"].keys())


# Checks that every regime used by an experiment is defined in regimes.yaml.
def test_all_experiment_regimes_exist_in_regimes_config():
    experiments_config = _load_yaml("experiments.yaml")
    regimes_config = _load_yaml("regimes.yaml")

    valid_regimes = set(regimes_config["regimes"].keys())

    for name, experiment in experiments_config["experiments"].items():
        regimes = set(experiment["regimes"])
        assert regimes.issubset(valid_regimes), name


# Checks that terminal-law methods are not accidentally treated as core path methods.
def test_terminal_only_methods_are_not_core_methods():
    config = _load_yaml("experiments.yaml")

    core = set(config["methods"]["core"])
    terminal_only = set(config["methods"]["terminal_only"])

    assert "Exact" in terminal_only
    assert "ChoiKwok" in terminal_only
    assert "Exact" not in core
    assert "ChoiKwok" not in core


# Checks that the strong-error reference grid is finer than every coarse grid.
def test_strong_error_reference_grid_is_finer_than_coarse_grids():
    config = _load_yaml("experiments.yaml")

    strong_error = config["time_grids"]["strong_error"]

    assert strong_error["reference_n_steps"] > max(strong_error["coarse_n_steps"])