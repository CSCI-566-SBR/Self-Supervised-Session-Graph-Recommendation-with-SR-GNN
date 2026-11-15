"""Tests for the standalone results collector."""

from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "reporting"
    / "collect_results.py"
)

SPEC = importlib.util.spec_from_file_location(
    "reporting_collector",
    SCRIPT_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import collector: {SCRIPT_PATH}")

collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)


class TestReportingCollector(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)

        self.root = Path(temporary.name)
        self.experiments = self.root / "experiments"
        self.experiments.mkdir()
        self.output = self.root / "reports" / "results.csv"

    def make_result(
        self,
        name: str,
        *,
        model: str = "srgnn",
        seed: int = 2028,
        dataset: str = "diginetica",
        learning_rate: float = 0.001,
        legacy: bool = False,
    ) -> Path:
        directory = self.experiments / name
        directory.mkdir()

        config = {
            "dataset": dataset,
            "seed": seed,
            "learning_rate": learning_rate,
        }

        data = {
            "experiment": name,
            "dataset": dataset,
            "seed": seed,
            "config": config,
            "test": {
                "examples": 10,
                "recall@20": 0.5,
                "mrr@20": 0.2,
            },
            "history": [{"epoch": 1}],
        }

        if legacy:
            config["gnn_steps"] = 1
        else:
            data["model_type"] = model

        path = directory / "results.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def read_csv(self) -> list[dict[str, str]]:
        with self.output.open(
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            return list(csv.DictReader(handle))

    def test_collects_all_runs_without_filters(self) -> None:
        self.make_result("srgnn_run")
        self.make_result("sasrec_run", model="sasrec", seed=2029)

        count = collector.collect_results(
            self.experiments,
            self.output,
        )

        self.assertEqual(count, 2)
        self.assertEqual(
            {row["experiment"] for row in self.read_csv()},
            {"srgnn_run", "sasrec_run"},
        )

    def test_each_filter_selects_correct_rows(self) -> None:
        self.make_result("matching")
        self.make_result("different_model", model="sasrec")
        self.make_result("different_seed", seed=2029)
        self.make_result("different_dataset", dataset="retailrocket")

        cases = [
            (
                {"models": ["SRGNN"]},
                {"matching", "different_seed", "different_dataset"},
            ),
            (
                {"seeds": [2028]},
                {"matching", "different_model", "different_dataset"},
            ),
            (
                {"datasets": ["DIGINETICA"]},
                {"matching", "different_model", "different_seed"},
            ),
        ]

        for index, (filters, expected) in enumerate(cases):
            with self.subTest(filters=filters):
                self.output = self.root / f"filtered_{index}.csv"

                count = collector.collect_results(
                    self.experiments,
                    self.output,
                    **filters,
                )

                self.assertEqual(count, len(expected))
                self.assertEqual(
                    {row["experiment"] for row in self.read_csv()},
                    expected,
                )

    def test_combined_filters_must_all_match(self) -> None:
        self.make_result("matching")
        self.make_result("different_model", model="sasrec")
        self.make_result("different_seed", seed=2029)
        self.make_result("different_dataset", dataset="retailrocket")

        count = collector.collect_results(
            self.experiments,
            self.output,
            models=["srgnn"],
            seeds=[2028],
            datasets=["diginetica"],
        )

        self.assertEqual(count, 1)
        self.assertEqual(self.read_csv()[0]["experiment"], "matching")

    def test_multiple_filter_values_are_supported(self) -> None:
        self.make_result("first")
        self.make_result(
            "second",
            model="sasrec",
            seed=2029,
            dataset="retailrocket",
        )
        self.make_result("excluded", model="other")

        count = collector.collect_results(
            self.experiments,
            self.output,
            models=["srgnn", "sasrec"],
            seeds=[2028, 2029],
            datasets=["diginetica", "retailrocket"],
        )

        self.assertEqual(count, 2)
        self.assertEqual(
            {row["experiment"] for row in self.read_csv()},
            {"first", "second"},
        )

    def test_missing_metrics_remain_blank(self) -> None:
        self.make_result("run")

        collector.collect_results(self.experiments, self.output)
        row = self.read_csv()[0]

        self.assertEqual(row["test.ndcg@20"], "")
        self.assertEqual(row["test.loss"], "")
        self.assertEqual(row["peak_gpu_memory_bytes"], "")
        self.assertEqual(float(row["test.recall@20"]), 0.5)

    def test_existing_csv_is_not_overwritten(self) -> None:
        self.make_result("run")
        collector.collect_results(self.experiments, self.output)
        original_csv = self.output.read_bytes()

        with self.assertRaises(FileExistsError):
            collector.collect_results(self.experiments, self.output)

        self.assertEqual(self.output.read_bytes(), original_csv)

    def test_distinct_runs_with_same_model_and_seed_are_kept(self) -> None:
        self.make_result("run_a", learning_rate=0.001)
        self.make_result("run_b", learning_rate=0.0005)

        count = collector.collect_results(
            self.experiments,
            self.output,
            models=["srgnn"],
            seeds=[2028],
        )
        rows = self.read_csv()

        self.assertEqual(count, 2)
        self.assertEqual(len({row["source_file"] for row in rows}), 2)
        self.assertEqual(
            {float(row["config.learning_rate"]) for row in rows},
            {0.001, 0.0005},
        )

    def test_source_results_are_not_modified(self) -> None:
        paths = [
            self.make_result("first"),
            self.make_result("second", model="sasrec"),
        ]
        originals = {path: path.read_bytes() for path in paths}

        collector.collect_results(self.experiments, self.output)

        for path, original in originals.items():
            self.assertEqual(path.read_bytes(), original)

    def test_legacy_srgnn_result_is_supported(self) -> None:
        self.make_result("legacy", legacy=True)

        count = collector.collect_results(
            self.experiments,
            self.output,
            models=["srgnn"],
        )
        row = self.read_csv()[0]

        self.assertEqual(count, 1)
        self.assertEqual(row["model"], "srgnn")
        self.assertEqual(
            row["model_identification"],
            "legacy_config.gnn_steps",
        )

    def test_no_matching_results_does_not_create_csv(self) -> None:
        self.make_result("run")

        with self.assertRaisesRegex(ValueError, "no experiments match"):
            collector.collect_results(
                self.experiments,
                self.output,
                models=["sasrec"],
            )

        self.assertFalse(self.output.exists())

    def test_empty_directory_does_not_create_csv(self) -> None:
        with self.assertRaisesRegex(ValueError, "no results.json"):
            collector.collect_results(self.experiments, self.output)

        self.assertFalse(self.output.exists())

    def test_invalid_json_does_not_create_csv(self) -> None:
        path = self.make_result("invalid")
        path.write_text("{invalid json", encoding="utf-8")

        with self.assertRaises(ValueError):
            collector.collect_results(self.experiments, self.output)

        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()