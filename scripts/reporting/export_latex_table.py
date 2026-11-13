"""Export selected CSV columns as a LaTeX table.

Reads input without modifying it. Existing output files are never overwritten.
Requires only the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path


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
) -> str:
    """Represent missing values as N/A and optionally round numeric cells."""
    if value is None or not value.strip():
        return "N/A"

    text = value.strip()

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
) -> int:
    """Write a complete LaTeX table fragment with selected CSV columns."""
    if output_path.suffix.lower() != ".tex":
        raise ValueError("output filename must end in .tex")

    if output_path.exists():
        raise FileExistsError(
            f"output already exists: {output_path}; "
            "choose a different filename"
        )

    if not columns or len(columns) != len(set(columns)):
        raise ValueError("provide unique column names")

    if headers is not None and len(headers) != len(columns):
        raise ValueError("provide exactly one header per selected column")

    if not 0 <= decimals <= 12:
        raise ValueError("decimals must be between 0 and 12")

    if not re.fullmatch(r"[A-Za-z0-9:_-]+", label):
        raise ValueError(
            "label may contain only letters, digits, colon, underscore, and hyphen"
        )

    rounded = set(round_columns or [])
    if not rounded.issubset(columns):
        raise ValueError("round-columns must be among the selected columns")

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
    alignment = "".join("r" if name in rounded else "l" for name in columns)

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
        " & ".join(escape_latex(header) for header in display_headers) + r" \\",
        r"\midrule",
    ]

    for row in rows:
        cells = [
            format_cell(
                row.get(column),
                decimals,
                round_number=column in rounded,
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
    with output_path.open("x", encoding="utf-8") as handle:
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
        )
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")

    print(f"Exported {count} row(s).")
    print(f"LaTeX table saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()