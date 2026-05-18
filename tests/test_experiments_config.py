import yaml

from src.utils.io import config_path


# Takes a YAML filename and returns its contents as a Python dictionary.
def _load_yaml(filename: str) -> dict:
    with open(config_path(filename), encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_experiments_block_is_nonempty():
    config = _load_yaml("experiments.yaml")

    assert "experiments" in config
    assert isinstance(config["experiments"], dict)
    assert len(config["experiments"]) > 0

def test_each_experiment_has_required_fields():
    config = _load_yaml("experiments.yaml")

    for name, experiment in config["experiments"].items():
        assert experiment is not None, f"Experiment {name} is empty or incorrectly indented"
        assert "description" in experiment, f"Experiment {name} is missing description"
        assert "regimes" in experiment, f"Experiment {name} is missing regimes"
        assert "T" in experiment, f"Experiment {name} is missing T"
        assert "n_paths" in experiment, f"Experiment {name} is missing n_paths"

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