"""Command-line script to plot training and metric curves from one or more results.json files."""

from __future__ import annotations

import argparse
from pathlib import Path

from recsys.evaluation.plotting import plot_experiment_curves


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results",
        nargs="+",
        type=Path,
        help="Path(s) to results.json file(s) from experiment run(s).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("experiment_curves.png"),
        help="Output PNG file path (default: experiment_curves.png).",
    )
    parser.add_argument(
        "--title",
        "-t",
        type=str,
        default=None,
        help="Optional title for the top of the plot.",
    )
    parser.add_argument(
        "--cutoffs",
        "--cutoff",
        "-k",
        nargs="+",
        type=int,
        default=[10, 20],
        dest="cutoffs",
        help="Top-K cutoff(s) to plot for Recall, MRR, and NDCG (default: 10 20).",
    )
    args = parser.parse_args()

    out = plot_experiment_curves(
        results_paths=args.results,
        output_path=args.output,
        title=args.title,
        cutoffs=args.cutoffs,
    )
    print(f"Curves successfully generated and saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
