"""Train and evaluate the baseline from a TOML configuration."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from recsys.training.config import load_experiment_config
from recsys.training.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", type=int, help="Override the training seed")
    parser.add_argument("--output-dir", type=Path, help="Override the experiment directory")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    if args.seed is not None:
        if args.seed < 0:
            parser.error("--seed must be non-negative")
        config = replace(config, seed=args.seed)
    if args.output_dir is not None:
        config = replace(config, output_dir=args.output_dir)
    results = run_experiment(config)
    print(json.dumps({key: results[key] for key in ("best_validation", "test")}, indent=2))


if __name__ == "__main__":
    main()
