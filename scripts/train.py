"""Train and evaluate the SR-GNN baseline from a TOML configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from recsys.training.config import load_experiment_config
from recsys.training.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    results = run_experiment(config)
    print(json.dumps({key: results[key] for key in ("best_validation", "test")}, indent=2))


if __name__ == "__main__":
    main()
