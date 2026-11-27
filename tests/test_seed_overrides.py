"""CLI isolation checks without training or evaluating a model."""

import contextlib
import io
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import evaluate_robustness as evaluation
from scripts import train
from recsys.training.config import ExperimentConfig


class TrainingOverrideTests(unittest.TestCase):
    def setUp(self):
        self.original = ExperimentConfig(
            name="example", dataset="diginetica",
            processed_data_dir=Path("data/clean"),
            output_dir=Path("experiments/example_seed2028"), seed=2028,
        )

    def invoke(self, arguments):
        with (
            patch("sys.argv", ["train.py", "--config", "example.toml", *arguments]),
            patch.object(train, "load_experiment_config", return_value=self.original),
            patch.object(train, "run_experiment", return_value={
                "best_validation": {}, "test": {},
            }) as run,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            train.main()
        return run.call_args.args[0]

    def test_override_changes_only_seed_and_directory_without_mutating_base(self):
        actual = self.invoke([
            "--seed", "2024", "--output-dir", "experiments/example_seed2024",
        ])
        expected = self.original.as_serializable_dict()
        expected.update(seed=2024, output_dir="experiments/example_seed2024")
        self.assertEqual(actual.as_serializable_dict(), expected)
        self.assertEqual(self.original.seed, 2028)
        self.assertEqual(self.original.output_dir, Path("experiments/example_seed2028"))

    def test_no_overrides_preserves_original_config(self):
        self.assertIs(self.invoke([]), self.original)

    def test_zero_seed_is_an_override(self):
        self.assertEqual(self.invoke(["--seed", "0"]).seed, 0)

    def test_negative_seed_does_not_start_training(self):
        with (
            patch("sys.argv", ["train.py", "--config", "example.toml", "--seed", "-1"]),
            patch.object(train, "load_experiment_config", return_value=self.original),
            patch.object(train, "run_experiment") as run,
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            train.main()
        run.assert_not_called()


class EvaluationOverrideTests(unittest.TestCase):
    def invoke(self, document, arguments):
        with (
            patch("sys.argv", ["evaluate_robustness.py", "--config", "example.toml", *arguments]),
            patch.object(evaluation, "load_toml", return_value=document),
            patch.object(evaluation, "run_robustness_evaluation") as run,
        ):
            evaluation.main()
        return run.call_args.kwargs

    def test_training_toml_uses_overridden_directory_for_input_and_output(self):
        document = {
            "experiment": {"output_dir": "experiments/example_seed2028"},
            "data": {"processed_data_dir": "data/processed/retailrocket/clean"},
            "training": {"seed": 2028, "num_workers": 6},
        }
        actual = self.invoke(document, ["--output-dir", "experiments/example_seed2024"])
        self.assertEqual(actual["experiment_dir"], Path("experiments/example_seed2024"))
        self.assertEqual(actual["output_file"], Path("experiments/example_seed2024/robustness_results.json"))
        self.assertEqual(actual["checkpoint_name"], "best_model.pt")
        self.assertEqual(actual["data_root"], Path("data/processed/retailrocket"))
        self.assertEqual(document["experiment"]["output_dir"], "experiments/example_seed2028")

    def test_dedicated_eval_override_does_not_reuse_old_checkpoint_or_output(self):
        document = {
            "evaluation": {
                "experiment_dir": "experiments/example_seed2028",
                "output_file": "experiments/example_seed2028/old_robustness.json",
                "checkpoint": "/old/seed2028/model.pt",
            },
        }
        actual = self.invoke(document, ["--output-dir", "experiments/example_seed2027"])
        self.assertEqual(actual["experiment_dir"], Path("experiments/example_seed2027"))
        self.assertEqual(actual["output_file"], Path("experiments/example_seed2027/robustness_results.json"))
        self.assertEqual(actual["checkpoint_name"], "best_model.pt")

    def test_no_override_preserves_dedicated_eval_paths(self):
        document = {
            "evaluation": {
                "experiment_dir": "experiments/example_seed2028",
                "output_file": "reports/original.json",
                "checkpoint": "model.pt",
            },
        }
        actual = self.invoke(document, [])
        self.assertEqual(actual["experiment_dir"], Path("experiments/example_seed2028"))
        self.assertEqual(actual["output_file"], Path("reports/original.json"))
        self.assertEqual(actual["checkpoint_name"], "model.pt")
