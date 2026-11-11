import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from recsys.training.config import ExperimentConfig
from recsys.training.multi_seed import aggregate_seed_results, run_seed_experiments


def make_result(seed: int, recall: float, mrr: float) -> dict:
    checksums = {"train": "abc"}
    artifact = {"revision": "frozen", "checksums": checksums}
    return {
        "experiment": f"tiny_seed{seed}",
        "dataset": "synthetic",
        "dataset_artifact": artifact,
        "dataset_checksums": checksums,
        "git_commit": "commit",
        "parameter_count": 10,
        "seed": seed,
        "best_epoch": 2,
        "best_validation": {
            "recall@20": recall,
            "mrr@20": mrr,
            "examples": 4,
        },
        "test": {
            "recall@20": recall - 0.01,
            "mrr@20": mrr - 0.01,
            "examples": 5,
        },
        "runtime_seconds": 3.0,
        "config": {"output_dir": f"outputs/seed_{seed}"},
    }


class MultiSeedTests(unittest.TestCase):
    def test_aggregates_mean_and_sample_standard_deviation(self) -> None:
        results = [
            make_result(1, 0.4, 0.1),
            make_result(2, 0.5, 0.2),
            make_result(3, 0.6, 0.3),
        ]

        summary = aggregate_seed_results(results, metric_k=20)

        recall = summary["aggregate"]["best_validation"]["recall@20"]
        self.assertAlmostEqual(recall["mean"], 0.5)
        self.assertAlmostEqual(recall["sample_standard_deviation"], 0.1)
        self.assertEqual(recall["values_by_seed"]["2"], 0.5)
        self.assertEqual(summary["total_runtime_seconds"], 9.0)

    def test_rejects_incomparable_results(self) -> None:
        first = make_result(1, 0.4, 0.1)
        second = make_result(2, 0.5, 0.2)
        second["dataset_checksums"] = {"train": "different"}

        with self.assertRaisesRegex(ValueError, "dataset_checksums"):
            aggregate_seed_results([first, second], metric_k=20)

    def test_runs_each_seed_once_and_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "runs"
            config = ExperimentConfig(
                name="tiny",
                dataset="synthetic",
                processed_data_dir=root / "data",
                output_dir=root / "unused-single-run",
                deterministic=True,
            )
            seen_configs = []

            def fake_run(seed_config: ExperimentConfig) -> dict:
                seen_configs.append(seed_config)
                seed_config.output_dir.mkdir(parents=True)
                return make_result(seed_config.seed, 0.4, 0.1)

            with patch(
                "recsys.training.multi_seed.run_experiment", side_effect=fake_run
            ):
                summary = run_seed_experiments(config, [7, 8], output_root)

            self.assertEqual([item.seed for item in seen_configs], [7, 8])
            self.assertTrue(all(item.deterministic for item in seen_configs))
            self.assertEqual(seen_configs[0].output_dir, output_root / "seed_7")
            saved = json.loads(
                (output_root / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved["seeds"], [7, 8])
            self.assertEqual(summary["number_of_runs"], 2)

            with self.assertRaises(FileExistsError):
                run_seed_experiments(config, [7, 8], output_root)


if __name__ == "__main__":
    unittest.main()
