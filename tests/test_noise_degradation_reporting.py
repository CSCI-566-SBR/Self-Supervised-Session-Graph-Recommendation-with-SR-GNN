"""Tests for noise-degradation export; no real datasets are required."""

import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/reporting/summarize_noise_degradation.py"
SPEC = importlib.util.spec_from_file_location("noise_reporter_under_test", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import {SCRIPT}")
reporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporter)


class TestNoiseDegradationReporting(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "run/robustness_results.json"
        self.source.parent.mkdir()
        self.output = self.root / "generated/noise.csv"
        self.metrics = ["recall@20", "mrr@20"]
        self.data = {
            "clean": {"examples": 100, "recall@20": 0.5, "mrr@20": 0.2},
            "deletion_0.1": {"examples": 100, "recall@20": 0.45, "mrr@20": 0.18},
        }
        self.write_data(self.data)

    def write_data(self, data):
        self.source.write_text(json.dumps(data), encoding="utf-8")

    def export(self, paths=None, metrics=None):
        return reporter.summarize_noise_degradation(
            [self.source] if paths is None else paths,
            self.output,
            self.metrics if metrics is None else metrics,
        )

    def read_rows(self):
        with self.output.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_correct_drop_formulas_and_source_preservation(self):
        before = self.source.read_bytes()
        self.assertEqual(self.export(), 2)
        row = self.read_rows()[0]
        self.assertAlmostEqual(float(row["absolute_drop_percentage_points"]), 5)
        self.assertAlmostEqual(float(row["relative_degradation_pct"]), 10)
        self.assertEqual(row["clean_score"], "0.5")
        self.assertEqual(row["condition_examples"], "100")
        self.assertEqual(row["source_file"], self.source.resolve().as_posix())
        self.assertEqual(row["source_sha256"], hashlib.sha256(before).hexdigest())
        self.assertEqual(self.source.read_bytes(), before)

    def test_all_corruption_types_are_supported(self):
        for name in ["substitution_0.05", "mixed_0.2"]:
            self.data[name] = copy.deepcopy(self.data["deletion_0.1"])
        self.write_data(self.data)
        self.assertEqual(self.export(), 6)
        self.assertEqual(
            {row["corruption_type"] for row in self.read_rows()},
            {"deletion", "substitution", "mixed"},
        )

    def test_zero_clean_score_has_blank_relative_drop(self):
        self.data["clean"]["recall@20"] = 0
        self.data["deletion_0.1"]["recall@20"] = 0
        self.write_data(self.data)
        self.export()
        row = self.read_rows()[0]
        self.assertEqual(row["relative_degradation_pct"], "")
        self.assertEqual(float(row["absolute_drop_percentage_points"]), 0)

    def test_improvement_is_reported_as_negative_degradation(self):
        self.data["deletion_0.1"]["recall@20"] = 0.55
        self.write_data(self.data)
        self.export()
        row = self.read_rows()[0]
        self.assertAlmostEqual(float(row["absolute_drop_percentage_points"]), -5)
        self.assertAlmostEqual(float(row["relative_degradation_pct"]), -10)

    def test_companion_metadata_and_fingerprint_are_recorded(self):
        companion = self.source.parent / "results.json"
        companion.write_text(json.dumps({"model_type": "sasrec", "dataset": "diginetica",
                                         "seed": 2028}), encoding="utf-8")
        before = companion.read_bytes()
        self.export()
        row = self.read_rows()[0]
        self.assertEqual((row["model"], row["dataset"], row["seed"]),
                         ("sasrec", "diginetica", "2028"))
        self.assertEqual(row["metadata_sha256"], hashlib.sha256(before).hexdigest())
        self.assertEqual(companion.read_bytes(), before)

    def test_legacy_and_config_model_identification(self):
        companion = self.source.parent / "results.json"
        for config, expected in [({"gnn_steps": 1}, "srgnn"),
                                 ({"model_type": "sasrec"}, "sasrec")]:
            with self.subTest(config=config):
                companion.write_text(json.dumps({"config": config}), encoding="utf-8")
                self.assertEqual(reporter.rows_from_result(self.source, self.metrics)[0]["model"], expected)

    def test_absent_model_metadata_is_not_guessed_from_directory(self):
        self.export()
        self.assertEqual(self.read_rows()[0]["model"], "unknown")

    def test_repeated_path_is_deduplicated_but_distinct_runs_are_kept(self):
        other = self.root / "other/robustness_results.json"
        other.parent.mkdir()
        other.write_bytes(self.source.read_bytes())
        self.assertEqual(self.export(paths=[self.source, self.source, other]), 4)
        self.assertEqual(len({row["source_file"] for row in self.read_rows()}), 2)

    def test_invalid_scores_are_rejected_without_output(self):
        for section in ["clean", "deletion_0.1"]:
            for value in [True, None, "0.5", -0.1, 1.1, float("nan"), float("inf")]:
                with self.subTest(section=section, value=value):
                    data = copy.deepcopy(self.data)
                    data[section]["recall@20"] = value
                    self.write_data(data)
                    with self.assertRaises(ValueError):
                        self.export()
                    self.assertFalse(self.output.exists())

    def test_missing_metric_is_rejected(self):
        del self.data["deletion_0.1"]["mrr@20"]
        self.write_data(self.data)
        with self.assertRaises(ValueError):
            self.export()
        self.assertFalse(self.output.exists())

    def test_invalid_or_mismatched_example_counts_are_rejected(self):
        for value in [0, -1, True, 1.5, "100", None, 99]:
            with self.subTest(value=value):
                data = copy.deepcopy(self.data)
                data["deletion_0.1"]["examples"] = value
                self.write_data(data)
                with self.assertRaises(ValueError):
                    self.export()
                self.assertFalse(self.output.exists())

    def test_invalid_conditions_and_equivalent_ratios_are_rejected(self):
        cases = [
            {"clean": self.data["clean"]},
            {**self.data, "other_0.1": self.data["deletion_0.1"]},
            {**self.data, "deletion_0": self.data["deletion_0.1"]},
            {**self.data, "deletion_1.1": self.data["deletion_0.1"]},
            {**self.data, "deletion_0.10": self.data["deletion_0.1"]},
        ]
        for data in cases:
            with self.subTest(keys=list(data)):
                self.write_data(data)
                with self.assertRaises(ValueError):
                    self.export()
                self.assertFalse(self.output.exists())

    def test_invalid_json_and_object_schemas_are_rejected(self):
        for text in ["not json", "[]", "{}", '{"clean": []}',
                     json.dumps({**self.data, "deletion_0.1": []})]:
            with self.subTest(text=text):
                self.source.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.export()
                self.assertFalse(self.output.exists())

    def test_invalid_metrics_or_empty_inputs_are_rejected(self):
        for metrics in [[], ["recall@20", "recall@20"], ["loss"], ["mrr@0"]]:
            with self.subTest(metrics=metrics):
                with self.assertRaises(ValueError):
                    self.export(metrics=metrics)
        with self.assertRaises(ValueError):
            self.export(paths=[])
        self.assertFalse(self.output.exists())

    def test_missing_input_and_wrong_extension_are_rejected(self):
        with self.assertRaises(ValueError):
            self.export(paths=[self.root / "missing.json"])
        with self.assertRaises(ValueError):
            reporter.summarize_noise_degradation([self.source], self.root / "bad.txt", self.metrics)
        self.assertFalse(self.output.exists())

    def test_existing_output_is_preserved(self):
        self.output.parent.mkdir()
        self.output.write_text("keep me", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.export()
        self.assertEqual(self.output.read_text(encoding="utf-8"), "keep me")

    def test_later_invalid_file_prevents_partial_csv(self):
        invalid = self.root / "zzz-invalid.json"
        invalid.write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.export(paths=[self.source, invalid])
        self.assertFalse(self.output.exists())

    def test_cli_exports_and_refuses_overwrite(self):
        command = [sys.executable, str(SCRIPT), "--results", str(self.source),
                   "--metrics", *self.metrics, "--output", str(self.output)]
        first = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(first.returncode, 0, first.stderr)
        before = self.output.read_bytes()
        second = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(second.returncode, 0)
        self.assertIn("already exists", second.stderr)
        self.assertEqual(self.output.read_bytes(), before)


    def test_clean_model_type_used_without_companion(self):
        self.data["clean"]["model_type"] = "sasrec"
        self.write_data(self.data)

        self.export()

        self.assertEqual(self.read_rows()[0]["model"], "sasrec")

    def test_clean_model_type_used_when_companion_has_no_model(self):
        self.data["clean"]["model_type"] = "sasrec"
        self.write_data(self.data)

        companion = self.source.parent / "results.json"
        companion.write_text(
            json.dumps({"config": {"seed": 2028}}),
            encoding="utf-8",
        )

        self.export()

        self.assertEqual(self.read_rows()[0]["model"], "sasrec")

    def test_explicit_companion_model_takes_priority(self):
        self.data["clean"]["model_type"] = "sasrec"
        self.write_data(self.data)

        companion = self.source.parent / "results.json"
        companion.write_text(
            json.dumps({"model_type": "srgnn"}),
            encoding="utf-8",
        )

        self.export()

        self.assertEqual(self.read_rows()[0]["model"], "srgnn")


if __name__ == "__main__":
    unittest.main()