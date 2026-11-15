"""Tests for overwrite support across the three retained reporting commands."""

import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts/reporting"

SPEC = importlib.util.spec_from_file_location(
    "reporting_output_under_test",
    SCRIPT_DIR / "_output.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Cannot import reporting output helper")

output = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(output)


class TestReportingOverwrite(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

        run = self.root / "experiments/run"
        run.mkdir(parents=True)

        (run / "results.json").write_text(
            json.dumps({
                "model_type": "srgnn",
                "dataset": "diginetica",
                "seed": 2028,
                "test": {"recall@20": 0.5},
            }),
            encoding="utf-8",
        )

        summary = self.root / "summary.json"
        summary.write_text(
            json.dumps({
                "dataset": "diginetica",
                "protocol_version": "v1",
                "num_items": 10,
                "minimum_item_frequency": 5,
                "minimum_session_length": 2,
                "filtering_protocol": "global recursive filtering",
                "item_mapping_scope": "training_split_only",
                "prefix_target_generation": "all_nonempty_prefixes",
                "counts": {
                    name: 10
                    for name in [
                        "train_sessions",
                        "validation_sessions",
                        "test_sessions",
                        "train_examples",
                        "validation_examples",
                        "test_examples",
                    ]
                },
                "split_strategy": {
                    "type": "chronological",
                    "validation_definition": "previous week",
                    "test_definition": "last week",
                    "validation_start": "2026-01-01",
                    "test_start": "2026-01-08",
                },
            }),
            encoding="utf-8",
        )

        self.csv_path = self.root / "results.csv"
        row = {
            "source_file": str(run / "results.json"),
            "model": "srgnn",
            "dataset": "diginetica",
            "seed": "2028",
            "test.recall@20": "0.5",
            "runtime_seconds": "10",
            "peak_gpu_memory_bytes": "1048576",
        }

        with self.csv_path.open(
            "w", newline="", encoding="utf-8-sig"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)

        self.cases = [
            (
                "collect_results.py",
                ["--results-root", str(run.parent)],
                ".csv",
            ),
            (
                "generate_dataset_summary.py",
                ["--summaries", str(summary)],
                ".csv",
            ),
            (
                "export_latex_table.py",
                ["--csv", str(self.csv_path), "--columns", "model"],
                ".tex",
            ),
        ]

    def invoke(self, script, arguments, destination, *extra):
        return subprocess.run(
            [
                sys.executable,
                "-m",
                f"scripts.reporting.{Path(script).stem}",
                *arguments,
                "--output",
                str(destination),
                *extra,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

    def test_all_commands_refuse_overwrite_by_default(self):
        for script, arguments, suffix in self.cases:
            with self.subTest(script=script):
                destination = self.root / (script + suffix)
                destination.write_bytes(b"keep me")

                result = self.invoke(script, arguments, destination)

                self.assertNotEqual(result.returncode, 0)
                self.assertIn("already exists", result.stderr)
                self.assertEqual(destination.read_bytes(), b"keep me")

    def test_all_commands_allow_explicit_overwrite_and_preserve_inputs(self):
        originals = {
            path: path.read_bytes()
            for path in self.root.rglob("*")
            if path.is_file()
        }

        for script, arguments, suffix in self.cases:
            with self.subTest(script=script):
                destination = self.root / (script + suffix)
                destination.write_bytes(b"old output")

                result = self.invoke(
                    script, arguments, destination, "--overwrite"
                )

                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotEqual(
                    destination.read_bytes(), b"old output"
                )

        for path, content in originals.items():
            self.assertEqual(path.read_bytes(), content)

        self.assertEqual(list(self.root.glob(".*.tmp")), [])

    def test_all_commands_preserve_existing_outputs_on_invalid_input(self):
        for path in self.root.rglob("*.json"):
            path.write_text("not json", encoding="utf-8")

        self.csv_path.write_text(
            "wrong\nvalue\n",
            encoding="utf-8",
        )

        for script, arguments, suffix in self.cases:
            with self.subTest(script=script):
                destination = self.root / (script + suffix)
                destination.write_bytes(b"keep me")

                result = self.invoke(
                    script, arguments, destination, "--overwrite"
                )

                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(destination.read_bytes(), b"keep me")

    def test_force_alias_is_supported_by_all_commands(self):
        for script, _, _ in self.cases:
            with self.subTest(script=script):
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        f"scripts.reporting.{Path(script).stem}",
                        "--help",
                    ],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                )

                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--overwrite", result.stdout)
                self.assertIn("-f", result.stdout)

        destination = self.root / "force.csv"
        destination.write_text("old", encoding="utf-8")
        script, arguments, _ = self.cases[0]

        result = self.invoke(script, arguments, destination, "-f")

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_helper_rejects_same_path_input_even_with_overwrite(self):
        before = self.csv_path.read_bytes()

        with self.assertRaisesRegex(ValueError, "input file"):
            with output.open_output(
                self.csv_path,
                overwrite=True,
                protected_paths=[self.csv_path],
            ):
                self.fail("Must not open an input as output")

        self.assertEqual(self.csv_path.read_bytes(), before)

    def test_helper_rejects_hardlink_input_alias(self):
        alias = self.root / "alias.csv"

        try:
            alias.hardlink_to(self.csv_path)
        except OSError:
            self.skipTest("Hard links unavailable on this filesystem")

        with self.assertRaises(ValueError):
            output.check_output(
                alias,
                overwrite=True,
                protected_paths=[self.csv_path],
            )

    def test_helper_failed_write_preserves_old_file_and_cleans_temporary(self):
        destination = self.root / "table.csv"
        destination.write_bytes(b"old")

        with self.assertRaises(RuntimeError):
            with output.open_output(
                destination, overwrite=True
            ) as handle:
                handle.write("partial")
                raise RuntimeError("Simulated failed write")

        self.assertEqual(destination.read_bytes(), b"old")
        self.assertEqual(list(self.root.glob(".*.tmp")), [])

    def test_helper_failed_replace_preserves_old_file_and_cleans_temporary(self):
        destination = self.root / "table.csv"
        destination.write_bytes(b"old")

        with patch.object(
            output.os,
            "replace",
            side_effect=PermissionError("locked"),
        ):
            with self.assertRaises(PermissionError):
                with output.open_output(
                    destination, overwrite=True
                ) as handle:
                    handle.write("new")

        self.assertEqual(destination.read_bytes(), b"old")
        self.assertEqual(list(self.root.glob(".*.tmp")), [])

    def test_helper_failed_new_output_is_removed(self):
        destination = self.root / "table.csv"

        with self.assertRaises(RuntimeError):
            with output.open_output(destination) as handle:
                handle.write("partial")
                raise RuntimeError("Failed")

        self.assertFalse(destination.exists())

    def test_latex_cli_rejects_input_output_alias(self):
        # Input protection must work regardless of filename extension.
        source = self.root / "input.tex"
        source.write_text("model\nsrgnn\n", encoding="utf-8")
        before = source.read_bytes()

        result = self.invoke(
            "export_latex_table.py",
            ["--csv", str(source), "--columns", "model"],
            source,
            "--overwrite",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("input file", result.stderr)
        self.assertEqual(source.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()