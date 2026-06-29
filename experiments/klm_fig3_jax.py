import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from klm_jax.fig3 import run_fig3_experiment


def get_args():
    parser = argparse.ArgumentParser(description="Run the KLM JAX Fig. 3 experiment.")

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a small fast version for testing.",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments.yaml"),
        help="Path to the experiment configuration file.",
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs/klm_fig3_jax"),
        help="Folder where results will be saved.",
    )

    return parser.parse_args()


def load_config(path, quick):
    with path.open("r", encoding="utf-8") as file:
        all_configs = yaml.safe_load(file)

    run_name = "quick" if quick else "full"
    return all_configs["klm_fig3_jax"][run_name]


def save_csv(rows, path):
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = get_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config, args.quick)
    results = run_fig3_experiment(config, outdir=args.outdir)

    with (args.outdir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    save_csv(results, args.outdir / "results.csv")

    print("Finished KLM Fig. 3 run.")
    print(f"Results saved in: {args.outdir}")


if __name__ == "__main__":
    main()