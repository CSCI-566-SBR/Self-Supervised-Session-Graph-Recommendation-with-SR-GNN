"""Tests for the standalone LaTeX exporter; uses temporary files only."""

import csv
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/reporting/export_latex_table.py"
SPEC = importlib.util.spec_from_file_location("latex_reporter_under_test", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import {SCRIPT}")
reporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporter)


class TestExportLatexTable(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "results.csv"
        self.output = self.root / "generated/table.tex"
        self.write_csv([["sr_gnn", "0.4950588469"], ["sasrec", ""]])

    def write_csv(self, rows):
        with self.source.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["model", "test.recall@20"])
            writer.writerows(rows)

    def export(self, **kwargs):
        return reporter.export_latex_table(
            self.source, self.output, ["model", "test.recall@20"], **kwargs
        )

    def test_exports_table_and_preserves_input(self):
        before = self.source.read_bytes()
        self.assertEqual(self.export(), 2)
        table = self.output.read_text(encoding="utf-8")
        for fragment in [r"\begin{table}", r"\toprule", r"\bottomrule", r"\end{table}"]:
            self.assertIn(fragment, table)
        self.assertIn(r"sr\_gnn", table)
        self.assertIn("N/A", table)
        self.assertEqual(self.source.read_bytes(), before)

    def test_escapes_all_special_characters(self):
        cases = {
            "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
            "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
            "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
        }
        for source, expected in cases.items():
            with self.subTest(character=source):
                self.assertEqual(reporter.escape_latex(source), expected)
        self.assertEqual(reporter.escape_latex("a\r\nb"), "a  b")

    def test_custom_headers_and_caption_are_escaped(self):
        self.export(headers=["Model_name", "$Recall$"], caption="A&B 50%")
        table = self.output.read_text(encoding="utf-8")
        self.assertIn(r"Model\_name", table)
        self.assertIn(r"\$Recall\$", table)
        self.assertIn(r"\caption{A\&B 50\%}", table)

    def test_rounding_does_not_scale_fractions(self):
        self.export(round_columns=["test.recall@20"], decimals=4)
        table = self.output.read_text(encoding="utf-8")
        self.assertIn("0.4951", table)
        self.assertNotIn("49.51", table)
        self.assertIn(r"\begin{tabular}{lr}", table)

    def test_unrounded_values_are_preserved(self):
        self.export()
        self.assertIn("0.4950588469", self.output.read_text(encoding="utf-8"))

    def test_missing_cells_become_na(self):
        for value in [None, "", "   "]:
            with self.subTest(value=value):
                self.assertEqual(
                    reporter.format_cell(value, 4, round_number=True), "N/A"
                )

    def test_existing_output_is_preserved(self):
        self.output.parent.mkdir()
        self.output.write_text("keep me", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.export()
        self.assertEqual(self.output.read_text(encoding="utf-8"), "keep me")

    def test_invalid_options_do_not_create_output(self):
        cases = [
            {"headers": ["one"]}, {"round_columns": ["absent"]},
            {"decimals": -1}, {"decimals": 13}, {"label": "bad{label}"},
        ]
        for options in cases:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    self.export(**options)
                self.assertFalse(self.output.exists())

    def test_invalid_selected_columns_are_rejected(self):
        for columns in [[], ["model", "model"], ["absent"]]:
            with self.subTest(columns=columns):
                with self.assertRaises(ValueError):
                    reporter.export_latex_table(self.source, self.output, columns)
                self.assertFalse(self.output.exists())

    def test_empty_csv_is_rejected(self):
        self.write_csv([])
        with self.assertRaises(ValueError):
            self.export()
        self.assertFalse(self.output.exists())

    def test_missing_input_is_rejected(self):
        with self.assertRaises(OSError):
            reporter.export_latex_table(self.root / "missing.csv", self.output, ["model"])
        self.assertFalse(self.output.exists())

    def test_wrong_output_extension_is_rejected(self):
        with self.assertRaises(ValueError):
            reporter.export_latex_table(self.source, self.root / "table.csv", ["model"])

    def test_nonfinite_rounded_values_are_rejected_before_writing(self):
        for value in ["NaN", "Infinity", "-Infinity"]:
            with self.subTest(value=value):
                self.write_csv([["model", value]])
                with self.assertRaises(ValueError):
                    self.export(round_columns=["test.recall@20"])
                self.assertFalse(self.output.exists())

    def test_cli_exports_and_refuses_overwrite(self):
        command = [sys.executable, str(SCRIPT), "--csv", str(self.source),
                   "--output", str(self.output), "--columns", "model"]
        first = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(first.returncode, 0, first.stderr)
        before = self.output.read_bytes()
        second = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(second.returncode, 0)
        self.assertIn("already exists", second.stderr)
        self.assertEqual(self.output.read_bytes(), before)

    def test_selected_percentage_columns_are_scaled_and_marked(self):
        self.export(
            percent_columns=["test.recall@20"],
            decimals=2,
        )

        table = self.output.read_text(encoding="utf-8")

        self.assertIn(r"49.51\%", table)
        self.assertIn(r"\begin{tabular}{lr}", table)
        self.assertIn("N/A", table)

    def test_percent_formatting_leaves_unselected_columns_unchanged(self):
        self.write_csv([["0.5", "0.4951"]])

        self.export(
            percent_columns=["test.recall@20"],
            decimals=2,
        )

        table = self.output.read_text(encoding="utf-8")
        self.assertIn(r"0.5 & 49.51\%", table)

    def test_percentage_endpoints_and_zero_decimals(self):
        self.write_csv([["a", "0"], ["b", "1"]])

        self.export(
            percent_columns=["test.recall@20"],
            decimals=0,
        )

        table = self.output.read_text(encoding="utf-8")
        self.assertIn(r"a & 0\%", table)
        self.assertIn(r"b & 100\%", table)

    def test_percentage_and_rounding_do_not_double_scale(self):
        self.export(
            percent_columns=["test.recall@20"],
            round_columns=["test.recall@20"],
            decimals=2,
        )

        table = self.output.read_text(encoding="utf-8")
        self.assertIn(r"49.51\%", table)

    def test_invalid_percentage_values_preserve_existing_output(self):
        self.output.parent.mkdir()
        self.output.write_text("old table", encoding="utf-8")

        for value in ["NaN", "Infinity", "text", "-0.1", "49.51"]:
            with self.subTest(value=value):
                self.write_csv([["a", value]])

                with self.assertRaises(ValueError):
                    self.export(
                        percent_columns=["test.recall@20"],
                        overwrite=True,
                    )

                self.assertEqual(
                    self.output.read_text(encoding="utf-8"),
                    "old table",
                )

    def test_invalid_percentage_columns_are_rejected(self):
        for columns in [["missing"], ["model", "model"]]:
            with self.subTest(columns=columns):
                with self.assertRaises(ValueError):
                    self.export(percent_columns=columns)

                self.assertFalse(self.output.exists())

    def test_raw_headers_are_opt_in_and_cells_still_escaped(self):
        self.export(
            headers=[r"$Model_i$", r"$Recall@20$"],
            raw_headers=True,
            caption="A&B",
        )

        table = self.output.read_text(encoding="utf-8")

        self.assertIn(r"$Model_i$ & $Recall@20$", table)
        self.assertIn(r"sr\_gnn", table)
        self.assertIn(r"\caption{A\&B}", table)

    def test_raw_headers_require_explicit_headers(self):
        with self.assertRaises(ValueError):
            self.export(raw_headers=True)

        self.assertFalse(self.output.exists())

    def test_cli_percentage_raw_headers_and_overwrite(self):
        self.output.parent.mkdir()
        self.output.write_text("old table", encoding="utf-8")
        before = self.source.read_bytes()

        command = [
            sys.executable,
            str(SCRIPT),
            "--csv", str(self.source),
            "--output", str(self.output),
            "--columns", "model", "test.recall@20",
            "--headers", "$Model$", "$Recall@20$",
            "--raw-headers",
            "--percent-columns", "test.recall@20",
            "--decimals", "2",
            "--overwrite",
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

        table = self.output.read_text(encoding="utf-8")
        self.assertIn(r"49.51\%", table)
        self.assertIn("$Recall@20$", table)
        self.assertEqual(self.source.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()