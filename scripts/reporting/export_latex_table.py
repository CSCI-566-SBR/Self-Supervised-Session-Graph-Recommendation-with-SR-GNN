"""Export selected CSV columns as a LaTeX table.

Reads input without modifying it. Existing output files are not overwritten unless --overwrite is supplied.
Requires only the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import re
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from scripts.reporting._output import check_output, open_output


LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def escape_latex(value: str) -> str:
    """Escape CSV text so it cannot accidentally become LaTeX commands."""
    return "".join(
        LATEX_ESCAPES.get(character, character)
        for character in value.replace("\r", " ").replace("\n", " ")
    )


def format_cell(
    value: str | None,
    decimals: int | None,
    *,
    round_number: bool,
    percentage: bool = False,
) -> str:
    """Represent missing values as N/A and optionally round numeric cells."""
    if value is None or not value.strip():
        return "N/A"

    text = value.strip()

    if percentage:
        try:
            number = Decimal(text)
        except InvalidOperation as error:
            raise ValueError(
                f"percentage cell must be a numeric fraction: {text}"
            ) from error

        if not number.is_finite() or not 0 <= number <= 1:
            raise ValueError(
                "percentage cell must be a finite fraction "
                f"in [0, 1]: {text}"
            )

        with localcontext() as context:
            context.prec = max(
                28,
                len(number.as_tuple().digits) + 2,
            )
            scaled = number * 100

        places = decimals if decimals is not None else 4
        return f"{scaled:.{places}f}" + r"\%"

    if round_number and decimals is not None:
        try:
            number = Decimal(text)
        except InvalidOperation:
            pass
        else:
            if not number.is_finite():
                raise ValueError(f"non-finite numeric value: {text}")
            return escape_latex(f"{number:.{decimals}f}")

    return escape_latex(text)


def export_latex_table(
    csv_path: Path,
    output_path: Path,
    columns: list[str],
    *,
    headers: list[str] | None = None,
    round_columns: list[str] | None = None,
    decimals: int = 4,
    caption: str = "Experiment results",
    label: str = "tab:results",
    overwrite: bool = False,
    percent_columns: list[str] | None = None,
    raw_headers: bool = False,
) -> int:
    """Write a complete LaTeX table fragment with selected CSV columns."""
    if output_path.suffix.lower() != ".tex":
        raise ValueError("output filename must end in .tex")

    check_output(
        output_path,
        overwrite=overwrite,
        protected_paths=(csv_path,),
    )

    if not columns or len(columns) != len(set(columns)):
        raise ValueError("provide unique column names")

    if headers is not None and len(headers) != len(columns):
        raise ValueError("provide exactly one header per selected column")

    if raw_headers and headers is None:
        raise ValueError("--raw-headers requires explicit --headers")

    if not 0 <= decimals <= 12:
        raise ValueError("decimals must be between 0 and 12")

    if not re.fullmatch(r"[A-Za-z0-9:_-]+", label):
        raise ValueError(
            "label may contain only letters, digits, colon, underscore, and hyphen"
        )

    rounded = set(round_columns or [])
    if not rounded.issubset(columns):
        raise ValueError("round-columns must be among the selected columns")

    percentages = set(percent_columns or [])
    if len(percent_columns or []) != len(percentages):
        raise ValueError("percent-columns must not contain duplicates")
    if not percentages.issubset(columns):
        raise ValueError(
            "percent-columns must be among the selected columns"
        )

    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        available = set(reader.fieldnames or [])
        missing = set(columns) - available

        if missing:
            raise ValueError(
                "CSV columns not found: " + ", ".join(sorted(missing))
            )

        rows = list(reader)

    if not rows:
        raise ValueError("CSV contains no data rows")

    # Format all content before creating the output file.
    display_headers = headers if headers is not None else columns
    alignment = "".join(
        "r" if name in rounded | percentages else "l"
        for name in columns
    )

    lines = [
        "% Generated from CSV; edit the source data or exporter, not this table.",
        "% Requires \\usepackage{booktabs} in the document preamble.",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{" + escape_latex(caption) + "}",
        r"\label{" + label + "}",
        r"\begin{tabular}{" + alignment + "}",
        r"\toprule",
        " & ".join(
            header if raw_headers else escape_latex(header)
            for header in display_headers
        ) + r" \\",
        r"\midrule",
    ]

    for row in rows:
        cells = [
            format_cell(
                row.get(column),
                decimals,
                round_number=column in rounded,
                percentage=column in percentages,
            )
            for column in columns
        ]
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open_output(
        output_path,
        overwrite=overwrite,
        protected_paths=(csv_path,),
        encoding="utf-8",
    ) as handle:
        handle.write("\n".join(lines))

    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--columns", nargs="+", required=True)
    parser.add_argument("--headers", nargs="+")
    parser.add_argument("--round-columns", nargs="+")
    parser.add_argument("--decimals", type=int, default=4)
    parser.add_argument("--caption", default="Experiment results")
    parser.add_argument("--label", default="tab:results")
    parser.add_argument(
        "--percent-columns",
        nargs="+",
        help="Selected fraction columns to display as percentages.",
    )
    parser.add_argument(
        "--raw-headers",
        action="store_true",
        help=(
            "Use trusted explicit headers as raw LaTeX; "
            "cells remain escaped."
        ),
    )
    parser.add_argument(
        "--overwrite",
        "-f",
        action="store_true",
        help="Intentionally replace an existing output file.",
    )
    args = parser.parse_args()

    try:
        count = export_latex_table(
            args.csv,
            args.output,
            args.columns,
            headers=args.headers,
            round_columns=args.round_columns,
            decimals=args.decimals,
            caption=args.caption,
            label=args.label,
            overwrite=args.overwrite,
            percent_columns=args.percent_columns,
            raw_headers=args.raw_headers,
        )
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")

    print(f"Exported {count} row(s).")
    print(f"LaTeX table saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()