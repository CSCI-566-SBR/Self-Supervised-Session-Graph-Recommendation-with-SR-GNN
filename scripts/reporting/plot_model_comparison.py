"""Plot final test metrics from the results collector's CSV.

Each bar represents one saved experiment run.
Runs are never automatically averaged or merged.
Existing output files are not overwritten unless --overwrite is supplied.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
try:
    from scripts.reporting._output import check_output, open_output
except ModuleNotFoundError as error:
    if error.name not in {
        "scripts",
        "scripts.reporting",
        "scripts.reporting._output",
    }:
        raise
    from _output import check_output, open_output


def read_rows(
    csv_path: Path,
    dataset: str,
    models: list[str] | None = None,
    seeds: list[int] | None = None,
) -> list[dict[str, str]]:
    """Read and filter rows from the collector's CSV."""
    selected_models = (
        {model.casefold() for model in models}
        if models else None
    )
    selected_seeds = (
        {str(seed) for seed in seeds}
        if seeds else None
    )

    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)

        required = {"dataset", "model", "seed", "source_file"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                "CSV is missing required columns; "
                "use the output of collect_results.py"
            )

        rows = []
        for row in reader:
            if row["dataset"].strip().casefold() != dataset.casefold():
                continue
            if (
                selected_models is not None
                and row["model"].strip().casefold() not in selected_models
            ):
                continue
            if (
                selected_seeds is not None
                and row["seed"] not in selected_seeds
            ):
                continue
            rows.append(row)

    if not rows:
        raise ValueError("no CSV rows match the requested filters")

    sources = [row["source_file"] for row in rows]
    if any(not source for source in sources):
        raise ValueError("each row must identify its source result file")
    if len(sources) != len(set(sources)):
        raise ValueError("CSV contains duplicate source result files")

    return rows


def check_evaluation_identity(rows: list[dict[str, str]]) -> None:
    """Require matching test data and item mapping for multi-run charts."""
    if len(rows) <= 1:
        return

    for artifact in ("test", "item_mapping"):
        fingerprints = [
            row.get(f"dataset_artifact.checksums.{artifact}")
            or row.get(f"dataset_checksums.{artifact}")
            or ""
            for row in rows
        ]

        if any(not fingerprint for fingerprint in fingerprints):
            raise ValueError(
                f"cannot verify matching {artifact} artifacts: "
                "a checksum is missing"
            )

        if len(set(fingerprints)) != 1:
            raise ValueError(
                f"selected runs use different {artifact} artifacts; "
                "plot them separately"
            )


def plot_comparison(
    rows: list[dict[str, str]],
    metrics: list[str],
    output_path: Path,
    dataset: str,
    *,
    overwrite: bool = False,
    protected_paths: tuple[Path, ...] = (),
) -> Path:
    """Plot one panel per metric without substituting zero for missing data."""
    if not metrics:
        raise ValueError("provide at least one metric")
    if len(metrics) != len(set(metrics)):
        raise ValueError("metric names must not be repeated")
    if output_path.suffix.lower() != ".png":
        raise ValueError("output filename must end in .png")

    protected_paths = tuple(protected_paths) + tuple(
        Path(row["source_file"])
        for row in rows
        if row.get("source_file")
    )

    check_output(
        output_path,
        overwrite=overwrite,
        protected_paths=protected_paths,
    )

    check_evaluation_identity(rows)

    # Validate values before creating an output file.
    values_by_metric: dict[str, list[float | None]] = {}

    for metric in metrics:
        if not metric.startswith(("recall@", "mrr@", "ndcg@")):
            raise ValueError(
                f"unsupported metric: {metric}; "
                "use recall@K, mrr@K, or ndcg@K"
            )

        values: list[float | None] = []
        for row in rows:
            text = row.get(f"test.{metric}", "")
            if text is None or not text.strip():
                values.append(None)
                continue

            value = float(text)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(
                    f"invalid {metric} value for {row.get('experiment')}"
                )
            values.append(value)

        if all(value is None for value in values):
            raise ValueError(
                f"no selected run contains test.{metric}; "
                "choose a metric available in the saved results"
            )

        values_by_metric[metric] = values

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    model_names = [row["model"].strip().casefold() or "unknown" for row in rows]
    unique_models = sorted(set(model_names))
    palette = plt.get_cmap("tab10" if len(unique_models) <= 10 else "hsv")
    model_colors = {
        model: palette(index if len(unique_models) <= 10 else index / len(unique_models))
        for index, model in enumerate(unique_models)
    }

    labels = [
        (
            f"{index + 1}. {row['model']}\n"
            f"seed {row['seed']}\n"
            f"{row.get('experiment') or 'unnamed experiment'}"
        )
        for index, row in enumerate(rows)
    ]

    figure, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(max(6, len(rows) * 2.5) * len(metrics), 5),
        squeeze=False,
    )

    try:
        for axis, metric in zip(axes[0], metrics):
            values = values_by_metric[metric]

            for index, value in enumerate(values):
                if value is None:
                    axis.text(
                        index,
                        0.02,
                        "N/A",
                        ha="center",
                        transform=axis.get_xaxis_transform(),
                    )
                    continue

                axis.bar(index, value, color=model_colors[model_names[index]])
                axis.text(
                    index,
                    value + 0.015,
                    f"{value:.4f}",
                    ha="center",
                    fontsize=9,
                )

            axis.set_xticks(range(len(rows)), labels=labels)
            axis.set_xlim(-0.6, len(rows) - 0.4)
            axis.set_ylim(0, 1.08)
            axis.set_ylabel("Score (fraction)")
            axis.set_title(f"Test {metric.upper()}")
            axis.grid(axis="y", alpha=0.25)
            axis.set_axisbelow(True)

        figure.suptitle(
            f"{dataset}: final test results — one bar per run"
        )
        figure.legend(
            handles=[Patch(facecolor=model_colors[model], label=model)
                     for model in unique_models],
            title="Model",
            loc="upper center",
            bbox_to_anchor=(0.5, 0.94),
            ncol=min(len(unique_models), 5),
        )
        legend_rows = (len(unique_models) + 4) // 5
        figure.tight_layout(rect=(0, 0, 1, max(0.5, 0.86 - 0.05 * (legend_rows - 1))))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Exclusive creation protects existing charts.
        with open_output(
            output_path,
            overwrite=overwrite,
            protected_paths=protected_paths,
            binary=True,
        ) as handle:
            figure.savefig(
                handle,
                format="png",
                dpi=200,
                bbox_inches="tight",
            )
    finally:
        plt.close(figure)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", nargs="+", dest="models")
    parser.add_argument("--seed", nargs="+", type=int, dest="seeds")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["recall@20", "mrr@20"],
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--overwrite",
        "-f",
        action="store_true",
        help="Intentionally replace an existing output file.",
    )
    args = parser.parse_args()

    try:
        rows = read_rows(
            args.csv,
            args.dataset,
            args.models,
            args.seeds,
        )
        output = plot_comparison(
            rows,
            args.metrics,
            args.output,
            args.dataset,
            overwrite=args.overwrite,
            protected_paths=(args.csv,),
        )
    except (OSError, ValueError, ImportError) as error:
        parser.exit(1, f"Error: {error}\n")

    print(f"Plotted {len(rows)} experiment run(s).")
    print(f"Chart saved to: {output.resolve()}")


if __name__ == "__main__":
    main()
