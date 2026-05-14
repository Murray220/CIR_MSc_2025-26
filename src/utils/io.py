# Input output helpers for repo
# Functions are for automating file saves for figures and results

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create the directory if doesn't already exist
def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents = True,
               exist_ok = True)
    return path


# Return a path inside the figures directory
def figure_path(filename: str) -> Path:
    ensure_dir(FIGURES_DIR)
    return FIGURES_DIR / filename


# Return a path inside the results directory
def results_path(filename: str) -> Path:
    ensure_dir(RESULTS_DIR)
    return RESULTS_DIR / filename