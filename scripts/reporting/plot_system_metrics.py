"""Plot recorded runtime and peak GPU allocated memory from a results CSV.

Each bar represents one saved run. No averaging is performed.
Inputs are read-only, and existing output files are never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def read_rows(
    csv_path: Path,
    dataset: str,
    models: list[str] | None = None,
    seeds: list[int] | None = None,
) -> list[dict[str, str]]:
    """Select runs without combining different datasets."""
    selected_models = (
        {model.strip().casefold() for model in models}
        if models else None
    )
    selected_seeds = (
        {str(seed) for seed in seeds}
        if seeds else None
    )

    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"source_file", "dataset", "model", "seed"}

        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                "CSV is missing required columns; "
                "use the output of collect_results.py"
            )

        rows = []
        for row in reader:
            if row["dataset"].strip().casefold() != dataset.strip().casefold():
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


def read_nonnegative_number(
    row: dict[str, str],
    field: str,
) -> float | None:
    """Keep missing measurements distinct from actual zero measurements."""
    text = row.get(field)

    if text is None or not text.strip():
        return None

    try:
        value = float(text)
    except ValueError as error:
        raise ValueError(
            f"invalid {field} for {row.get('experiment', 'unnamed run')}"
        ) from error

    if not math.isfinite(value) or value < 0:
        raise ValueError(
            f"{field} must be finite and non-negative "
            f"for {row.get('experiment', 'unnamed run')}"
        )

    return value


def plot_system_metrics(
    rows: list[dict[str, str]],
    output_path: Path,
    dataset: str,
) -> Path:
    """Plot observed measurements without claiming hardware-normalized speed."""
    if not rows:
        raise ValueError("provide at least one experiment run")

    if output_path.suffix.lower() != ".png":
        raise ValueError("output filename must end in .png")

    if output_path.exists():
        raise FileExistsError(
            f"output already exists: {output_path}; "
            "choose a different chart filename"
        )

    runtimes = [
        read_nonnegative_number(row, "runtime_seconds")
        for row in rows
    ]
    memory_bytes = [
        read_nonnegative_number(row, "peak_gpu_memory_bytes")
        for row in rows
    ]
    memory_mib = [
        value / (1024 ** 2) if value is not None else None
        for value in memory_bytes
    ]

    if all(value is None for value in runtimes + memory_mib):
        raise ValueError(
            "no runtime or peak GPU memory measurements are available"
        )

    labels = [
        (
            f"{index + 1}. {row['model']}\n"
            f"seed {row['seed']} | {row.get('device') or 'device unknown'}\n"
            f"{row.get('experiment') or 'unnamed experiment'}"
        )
        for index, row in enumerate(rows)
    ]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(max(12, len(rows) * 4), 6),
    )

    panels = [
        (
            axes[0],
            runtimes,
            "Recorded run runtime",
            "Seconds",
            "#4477AA",
        ),
        (
            axes[1],
            memory_mib,
            "Peak GPU allocated memory",
            "MiB (1 MiB = 1,048,576 bytes)",
            "#228833",
        ),
    ]

    try:
        for axis, values, title, ylabel, color in panels:
            for index, value in enumerate(values):
                if value is None:
                    axis.text(
                        index,
                        0.03,
                        "N/A",
                        ha="center",
                        transform=axis.get_xaxis_transform(),
                    )
                else:
                    axis.bar(index, value, color=color)
                    axis.annotate(
                        f"{value:.2f}",
                        (index, value),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha="center",
                        fontsize=9,
                    )

            observed = [value for value in values if value is not None]
            upper_limit = max(observed, default=0)
            axis.set_ylim(0, upper_limit * 1.2 if upper_limit > 0 else 1)
            axis.set_xlim(-0.6, len(rows) - 0.4)
            axis.set_xticks(range(len(rows)), labels)
            axis.set_title(title)
            axis.set_ylabel(ylabel)
            axis.grid(axis="y", alpha=0.25)
            axis.set_axisbelow(True)

        figure.suptitle(
            f"{dataset}: recorded system metrics — one bar per run"
        )
        figure.text(
            0.5,
            0.02,
            "Descriptive only: hardware, epoch counts and run settings may differ.\n"
            "Exact hardware is not shown; CPU/CUDA labels do not establish equivalence. "
            "N/A means not recorded.\n"
            "Runtime reflects the runner's measured interval; GPU allocated memory "
            "is not total GPU usage or system RAM.",
            ha="center",
            fontsize=9,
        )
        figure.tight_layout(rect=(0, 0.15, 1, 0.94))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("xb") as handle:
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
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    try:
        rows = read_rows(
            args.csv,
            args.dataset,
            args.models,
            args.seeds,
        )
        output = plot_system_metrics(
            rows,
            args.output,
            args.dataset,
        )
    except (OSError, ValueError, ImportError) as error:
        parser.exit(1, f"Error: {error}\n")

    print(f"Plotted {len(rows)} experiment run(s).")
    print(f"Chart saved to: {output.resolve()}")


if __name__ == "__main__":
    main()