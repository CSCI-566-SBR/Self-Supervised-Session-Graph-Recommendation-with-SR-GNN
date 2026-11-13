"""Tests for final-test comparison plots using temporary files."""

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
    / "plot_model_comparison.py"
)

SPEC = importlib.util.spec_from_file_location(
    "model_comparison_reporter",
    SCRIPT_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import script: {SCRIPT_PATH}")

reporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporter)


class TestModelComparisonReporting(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)

        self.root = Path(temporary.name)
        self.csv_path = self.root / "results.csv"
        self.output = self.root / "charts" / "comparison.png"

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
            "test.recall@20": "0.5",
            "test.mrr@20": "0.2",
            "test.ndcg@20": "",
            "dataset_artifact.checksums.test": "same-test",
            "dataset_artifact.checksums.item_mapping": "same-mapping",
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

    def test_duplicate_source_rows_are_rejected(self) -> None:
        row = self.make_row()
        self.write_csv([row, dict(row)])

        with self.assertRaisesRegex(ValueError, "duplicate"):
            reporter.read_rows(self.csv_path, "diginetica")

    def test_no_matching_rows_are_rejected(self) -> None:
        self.write_csv([self.make_row()])

        with self.assertRaisesRegex(ValueError, "no CSV rows"):
            reporter.read_rows(self.csv_path, "retailrocket")

    def test_matching_fingerprints_are_accepted(self) -> None:
        rows = [
            self.make_row("first"),
            self.make_row("second", model="sasrec"),
        ]

        reporter.check_evaluation_identity(rows)

    def test_mismatched_fingerprints_are_rejected(self) -> None:
        for artifact in ("test", "item_mapping"):
            with self.subTest(artifact=artifact):
                rows = [
                    self.make_row("first"),
                    self.make_row("second"),
                ]
                rows[1][
                    f"dataset_artifact.checksums.{artifact}"
                ] = "different"

                with self.assertRaisesRegex(
                    ValueError,
                    f"different {artifact}",
                ):
                    reporter.check_evaluation_identity(rows)

    def test_missing_fingerprint_is_rejected_for_multiple_runs(self) -> None:
        rows = [
            self.make_row("first"),
            self.make_row("second"),
        ]
        del rows[1]["dataset_artifact.checksums.test"]

        with self.assertRaisesRegex(ValueError, "checksum is missing"):
            reporter.check_evaluation_identity(rows)

    def test_fallback_checksum_fields_are_supported(self) -> None:
        rows = [
            self.make_row("first"),
            self.make_row("second"),
        ]

        for row in rows:
            for artifact in ("test", "item_mapping"):
                row[f"dataset_checksums.{artifact}"] = row.pop(
                    f"dataset_artifact.checksums.{artifact}"
                )

        reporter.check_evaluation_identity(rows)

    def test_generates_png_without_modifying_csv(self) -> None:
        self.write_csv([self.make_row()])
        original_csv = self.csv_path.read_bytes()
        rows = reporter.read_rows(self.csv_path, "diginetica")

        result = reporter.plot_comparison(
            rows,
            ["recall@20", "mrr@20"],
            self.output,
            "diginetica",
        )

        self.assertEqual(result, self.output)
        self.assertTrue(
            self.output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        )
        self.assertEqual(self.csv_path.read_bytes(), original_csv)

    def test_existing_chart_is_not_overwritten(self) -> None:
        self.output.parent.mkdir()
        original = b"existing chart"
        self.output.write_bytes(original)

        with self.assertRaises(FileExistsError):
            reporter.plot_comparison(
                [self.make_row()],
                ["recall@20"],
                self.output,
                "diginetica",
            )

        self.assertEqual(self.output.read_bytes(), original)

    def test_metric_missing_from_every_run_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "no selected run"):
            reporter.plot_comparison(
                [self.make_row()],
                ["ndcg@20"],
                self.output,
                "diginetica",
            )

        self.assertFalse(self.output.exists())

    def test_partially_missing_metric_can_be_plotted(self) -> None:
        rows = [
            self.make_row("first"),
            self.make_row("second"),
        ]
        rows[1]["test.mrr@20"] = ""

        reporter.plot_comparison(
            rows,
            ["mrr@20"],
            self.output,
            "diginetica",
        )

        self.assertTrue(self.output.exists())

    def test_invalid_metric_values_are_rejected(self) -> None:
        for invalid in ("nan", "inf", "-0.1", "1.2", "invalid"):
            with self.subTest(value=invalid):
                row = self.make_row()
                row["test.mrr@20"] = invalid

                with self.assertRaises(ValueError):
                    reporter.plot_comparison(
                        [row],
                        ["mrr@20"],
                        self.output,
                        "diginetica",
                    )

                self.assertFalse(self.output.exists())

    def test_unsupported_metric_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported metric"):
            reporter.plot_comparison(
                [self.make_row()],
                ["loss"],
                self.output,
                "diginetica",
            )

        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()