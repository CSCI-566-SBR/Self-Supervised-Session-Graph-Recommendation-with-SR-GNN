"""Run a baseline configuration with multiple random seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from recsys.training.config import load_experiment_config
from recsys.training.multi_seed import run_seed_experiments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seeds", required=True, nargs="+", type=int)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    summary = run_seed_experiments(config, args.seeds, args.output_root)
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
