"""Tests for system-metric reporting using temporary files."""

from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "reporting"
    / "plot_system_metrics.py"
)

SPEC = importlib.util.spec_from_file_location(
    "system_metrics_reporter",
    SCRIPT_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import script: {SCRIPT_PATH}")

reporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporter)


class TestSystemMetricsReporting(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)

        self.root = Path(temporary.name)
        self.csv_path = self.root / "results.csv"
        self.output = self.root / "charts" / "system_metrics.png"

    def make_row(
        self,
        name: str = "run_a",
        model: str = "srgnn",
        seed: int = 2028,
        dataset: str = "diginetica",
    ) -> dict[str, str]:
        return {
            "source_file": f"/experiments/{name}/results.json",
            "experiment": name,
            "model": model,
            "seed": str(seed),
            "dataset": dataset,
            "device": "cuda",
            "runtime_seconds": "100",
            "peak_gpu_memory_bytes": "104857600",
        }

    def write_csv(self, rows: list[dict[str, str]]) -> None:
        columns = sorted({key for row in rows for key in row})

        with self.csv_path.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def test_filters_by_dataset_model_and_seed(self) -> None:
        self.write_csv([
            self.make_row("matching"),
            self.make_row("other_model", model="sasrec"),
            self.make_row("other_seed", seed=2029),
            self.make_row("other_dataset", dataset="retailrocket"),
        ])

        rows = reporter.read_rows(
            self.csv_path,
            dataset="DIGINETICA",
            models=["SRGNN"],
            seeds=[2028],
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["experiment"], "matching")

    def test_multiple_filter_values_are_supported(self) -> None:
        self.write_csv([
            self.make_row("first"),
            self.make_row("second", model="sasrec", seed=2029),
            self.make_row("excluded", model="other"),
        ])

        rows = reporter.read_rows(
            self.csv_path,
            dataset="diginetica",
            models=["srgnn", "sasrec"],
            seeds=[2028, 2029],
        )

        self.assertEqual(len(rows), 2)

    def test_duplicate_sources_are_rejected(self) -> None:
        row = self.make_row()
        self.write_csv([row, dict(row)])

        with self.assertRaisesRegex(ValueError, "duplicate"):
            reporter.read_rows(self.csv_path, "diginetica")

    def test_missing_source_is_rejected(self) -> None:
        row = self.make_row()
        row["source_file"] = ""
        self.write_csv([row])

        with self.assertRaisesRegex(ValueError, "source result file"):
            reporter.read_rows(self.csv_path, "diginetica")

    def test_missing_required_columns_are_rejected(self) -> None:
        self.write_csv([{"model": "srgnn"}])

        with self.assertRaisesRegex(ValueError, "required columns"):
            reporter.read_rows(self.csv_path, "diginetica")

    def test_no_matching_rows_are_rejected(self) -> None:
        self.write_csv([self.make_row()])

        with self.assertRaisesRegex(ValueError, "no CSV rows"):
            reporter.read_rows(self.csv_path, "retailrocket")

    def test_missing_measurement_returns_none(self) -> None:
        self.assertIsNone(
            reporter.read_nonnegative_number({}, "runtime_seconds")
        )
        self.assertIsNone(
            reporter.read_nonnegative_number(
                {"runtime_seconds": ""},
                "runtime_seconds",
            )
        )

    def test_zero_is_a_valid_measurement(self) -> None:
        value = reporter.read_nonnegative_number(
            {"runtime_seconds": "0"},
            "runtime_seconds",
        )

        self.assertEqual(value, 0.0)
        self.assertIsNotNone(value)

    def test_invalid_measurements_are_rejected(self) -> None:
        for field in ("runtime_seconds", "peak_gpu_memory_bytes"):
            for invalid in ("-1", "nan", "inf", "invalid"):
                with self.subTest(field=field, value=invalid):
                    row = self.make_row()
                    row[field] = invalid

                    with self.assertRaises(ValueError):
                        reporter.plot_system_metrics(
                            [row],
                            self.output,
                            "diginetica",
                        )

                    self.assertFalse(self.output.exists())

    def test_generates_png_without_modifying_csv(self) -> None:
        self.write_csv([self.make_row()])
        original_csv = self.csv_path.read_bytes()

        rows = reporter.read_rows(self.csv_path, "diginetica")
        result = reporter.plot_system_metrics(
            rows,
            self.output,
            "diginetica",
        )

        self.assertEqual(result, self.output)
        self.assertTrue(
            self.output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        )
        self.assertEqual(self.csv_path.read_bytes(), original_csv)

    def test_cpu_run_with_missing_gpu_memory_can_be_plotted(self) -> None:
        row = self.make_row()
        row["device"] = "cpu"
        row["peak_gpu_memory_bytes"] = ""

        reporter.plot_system_metrics(
            [row],
            self.output,
            "diginetica",
        )

        self.assertTrue(self.output.exists())

    def test_memory_only_run_can_be_plotted(self) -> None:
        row = self.make_row()
        row["runtime_seconds"] = ""

        reporter.plot_system_metrics(
            [row],
            self.output,
            "diginetica",
        )

        self.assertTrue(self.output.exists())

    def test_all_measurements_missing_is_rejected(self) -> None:
        row = self.make_row()
        row["runtime_seconds"] = ""
        row["peak_gpu_memory_bytes"] = ""

        with self.assertRaisesRegex(ValueError, "no runtime"):
            reporter.plot_system_metrics(
                [row],
                self.output,
                "diginetica",
            )

        self.assertFalse(self.output.exists())

    def test_existing_chart_is_not_overwritten(self) -> None:
        self.output.parent.mkdir()
        original = b"existing chart"
        self.output.write_bytes(original)

        with self.assertRaises(FileExistsError):
            reporter.plot_system_metrics(
                [self.make_row()],
                self.output,
                "diginetica",
            )

        self.assertEqual(self.output.read_bytes(), original)

    def test_empty_rows_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            reporter.plot_system_metrics(
                [],
                self.output,
                "diginetica",
            )

        self.assertFalse(self.output.exists())

    def test_non_png_output_is_rejected(self) -> None:
        output = self.root / "chart.csv"

        with self.assertRaisesRegex(ValueError, "end in .png"):
            reporter.plot_system_metrics(
                [self.make_row()],
                output,
                "diginetica",
            )

        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()