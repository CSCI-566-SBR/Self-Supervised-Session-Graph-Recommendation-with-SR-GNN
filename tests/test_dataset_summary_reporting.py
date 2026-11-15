"""Tests for dataset-summary reporting using temporary metadata."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "reporting"
    / "generate_dataset_summary.py"
)

SPEC = importlib.util.spec_from_file_location(
    "dataset_summary_reporter",
    SCRIPT_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import script: {SCRIPT_PATH}")

reporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporter)


class TestDatasetSummaryReporting(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "reports" / "dataset_summary.csv"

    def make_summary(self, dataset: str = "diginetica") -> Path:
        directory = self.root / dataset / "clean"
        directory.mkdir(parents=True, exist_ok=True)

        summary = {
            "dataset": dataset,
            "protocol_version": f"{dataset}_clean_v1",
            "num_items": 100,
            "counts": {
                "train_sessions": 80,
                "validation_sessions": 10,
                "test_sessions": 10,
                "train_examples": 160,
                "validation_examples": 20,
                "test_examples": 20,
            },
            "minimum_item_frequency": 5,
            "minimum_session_length": 2,
            "split_strategy": {
                "type": "chronological_session_end",
                "validation_definition": "7 days immediately before test",
                "test_definition": "final 7 days",
                "validation_start": "2026-01-01T00:00:00",
                "test_start": "2026-01-08T00:00:00",
            },
            "filtering_protocol": "global recursive filtering",
            "item_mapping_scope": "training_split_only",
            "prefix_target_generation": "all_nonempty_prefixes",
        }

        path = directory / "preprocessing_summary.json"
        self.write_summary(path, summary)
        return path

    def write_summary(self, path: Path, data: dict) -> None:
        path.write_text(json.dumps(data), encoding="utf-8")

    def read_csv(self) -> list[dict[str, str]]:
        with self.output.open(
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            return list(csv.DictReader(handle))

    def test_exports_expected_statistics(self) -> None:
        path = self.make_summary()

        count = reporter.generate_dataset_summary(
            [path],
            self.output,
        )
        rows = self.read_csv()

        self.assertEqual(count, 1)
        self.assertEqual(rows[0]["dataset"], "diginetica")
        self.assertEqual(rows[0]["num_items"], "100")
        self.assertEqual(rows[0]["train_sessions"], "80")
        self.assertEqual(rows[0]["train_examples"], "160")
        self.assertEqual(rows[0]["minimum_item_frequency"], "5")
        self.assertEqual(
            rows[0]["split_type"],
            "chronological_session_end",
        )

    def test_multiple_datasets_produce_separate_rows(self) -> None:
        paths = [
            self.make_summary("diginetica"),
            self.make_summary("retailrocket"),
        ]

        count = reporter.generate_dataset_summary(paths, self.output)

        self.assertEqual(count, 2)
        self.assertEqual(
            {row["dataset"] for row in self.read_csv()},
            {"diginetica", "retailrocket"},
        )

    def test_repeated_input_path_is_only_exported_once(self) -> None:
        path = self.make_summary()

        count = reporter.generate_dataset_summary(
            [path, path],
            self.output,
        )

        self.assertEqual(count, 1)
        self.assertEqual(len(self.read_csv()), 1)

    def test_source_file_and_fingerprint_are_recorded(self) -> None:
        path = self.make_summary()
        expected_hash = hashlib.sha256(path.read_bytes()).hexdigest()

        reporter.generate_dataset_summary([path], self.output)
        row = self.read_csv()[0]

        self.assertEqual(row["source_file"], path.resolve().as_posix())
        self.assertEqual(row["summary_sha256"], expected_hash)

    def test_source_metadata_is_unchanged(self) -> None:
        path = self.make_summary()
        original = path.read_bytes()

        reporter.generate_dataset_summary([path], self.output)

        self.assertEqual(path.read_bytes(), original)

    def test_existing_csv_is_not_overwritten(self) -> None:
        path = self.make_summary()
        reporter.generate_dataset_summary([path], self.output)
        original = self.output.read_bytes()

        with self.assertRaises(FileExistsError):
            reporter.generate_dataset_summary([path], self.output)

        self.assertEqual(self.output.read_bytes(), original)

    def test_noisy_summary_is_rejected(self) -> None:
        path = self.make_summary()
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["noise"] = {"corruption_type": "deletion", "ratio": 0.1}
        self.write_summary(path, summary)

        with self.assertRaisesRegex(ValueError, "clean summary"):
            reporter.generate_dataset_summary([path], self.output)

        self.assertFalse(self.output.exists())

    def test_missing_count_is_rejected(self) -> None:
        path = self.make_summary()
        summary = json.loads(path.read_text(encoding="utf-8"))
        del summary["counts"]["train_sessions"]
        self.write_summary(path, summary)

        with self.assertRaises(ValueError):
            reporter.generate_dataset_summary([path], self.output)

        self.assertFalse(self.output.exists())

    def test_invalid_counts_are_rejected(self) -> None:
        path = self.make_summary()
        summary = json.loads(path.read_text(encoding="utf-8"))

        for invalid in (-1, 0, True, "80"):
            with self.subTest(value=invalid):
                summary["counts"]["train_sessions"] = invalid
                self.write_summary(path, summary)

                with self.assertRaises(ValueError):
                    reporter.generate_dataset_summary([path], self.output)

                self.assertFalse(self.output.exists())

    def test_missing_file_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            reporter.generate_dataset_summary(
                [self.root / "missing.json"],
                self.output,
            )

        self.assertFalse(self.output.exists())

    def test_invalid_json_is_rejected(self) -> None:
        path = self.make_summary()
        path.write_text("{invalid json", encoding="utf-8")

        with self.assertRaises(ValueError):
            reporter.generate_dataset_summary([path], self.output)

        self.assertFalse(self.output.exists())

    def test_empty_input_list_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            reporter.generate_dataset_summary([], self.output)

        self.assertFalse(self.output.exists())

    def test_non_csv_output_is_rejected(self) -> None:
        path = self.make_summary()
        invalid_output = self.root / "output.json"

        with self.assertRaises(ValueError):
            reporter.generate_dataset_summary([path], invalid_output)

        self.assertFalse(invalid_output.exists())


if __name__ == "__main__":
    unittest.main()