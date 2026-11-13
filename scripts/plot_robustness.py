"""Command-line script to plot robustness degradation curves or bar charts from robustness_results.json."""

from __future__ import annotations

import argparse
from pathlib import Path

from recsys.evaluation.plotting import plot_robustness_bars, plot_robustness_curves


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results",
        nargs="+",
        type=Path,
        help="Path(s) to robustness_results.json file(s) or experiment director(ies).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("experiments/robustness_comparison.png"),
        help="Output PNG file path (default: experiments/robustness_comparison.png).",
    )
    parser.add_argument(
        "--metrics",
        "-m",
        nargs="+",
        type=str,
        default=["recall@20", "mrr@20"],
        help="Ranking metrics to plot (default: recall@20 mrr@20).",
    )
    parser.add_argument(
        "--labels",
        "-l",
        nargs="+",
        type=str,
        default=None,
        help="Optional custom labels for models (e.g. SASRec SR-GNN).",
    )
    parser.add_argument(
        "--style",
        "-s",
        choices=["curves", "bars"],
        default="curves",
        help="Plot style: 'curves' across corruption ratios (default) or 'bars' comparing conditions.",
    )
    parser.add_argument(
        "--title",
        "-t",
        type=str,
        default=None,
        help="Optional title for the figure.",
    )
    args = parser.parse_args()

    if args.style == "curves":
        out = plot_robustness_curves(
            results_paths=args.results,
            output_path=args.output,
            metrics=args.metrics,
            labels=args.labels,
            title=args.title,
        )
    else:
        out = plot_robustness_bars(
            results_paths=args.results,
            output_path=args.output,
            metric=args.metrics[0],
            labels=args.labels,
            title=args.title,
        )

    print(f"Robustness plot successfully generated and saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
