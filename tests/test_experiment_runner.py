import json
import pickle
import tempfile
import unittest
from pathlib import Path

from recsys.data.processed import load_processed_dataset
from recsys.training.config import ExperimentConfig, load_experiment_config
from recsys.training.runner import run_experiment


def write_synthetic_artifacts(root: Path) -> None:
    splits = {
        "train.txt": ([[1], [1, 2], [2, 3]], [2, 3, 4]),
        "validation.txt": ([[1, 2], [3]], [3, 4]),
        "test.txt": ([[2, 3], [4]], [4, 5]),
    }
    for filename, value in splits.items():
        with (root / filename).open("wb") as handle:
            pickle.dump(value, handle)
    with (root / "item_mapping.pkl").open("wb") as handle:
        pickle.dump({f"raw-{item}": item for item in range(1, 6)}, handle)
    (root / "preprocessing_summary.json").write_text(
        json.dumps(
            {
                "dataset": "synthetic",
                "filtering_protocol": "synthetic_test",
                "protocol_version": "synthetic_v1",
                "num_items": 5,
            }
        ),
        encoding="utf-8",
    )


class ExperimentConfigTests(unittest.TestCase):
    def test_loads_toml_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "experiment.toml"
            path.write_text(
                """
[experiment]
name = "tiny"
dataset = "synthetic"
output_dir = "outputs/tiny"

[data]
processed_data_dir = "data/tiny"
artifact_repository = "team/tiny"
artifact_revision = "abc123"
expected_protocol_version = "synthetic_v1"

[model]
hidden_size = 8
gnn_steps = 1
hybrid = true

[training]
batch_size = 2
max_epochs = 2
patience = 1
seed = 11
device = "cpu"
""".strip(),
                encoding="utf-8",
            )

            config = load_experiment_config(path)

            self.assertEqual(config.name, "tiny")
            self.assertEqual(config.hidden_size, 8)
            self.assertEqual(config.batch_size, 2)
            self.assertEqual(config.metric_k, 20)
            self.assertEqual(config.artifact_repository, "team/tiny")
            self.assertEqual(config.artifact_revision, "abc123")
            self.assertEqual(config.expected_protocol_version, "synthetic_v1")

    def test_rejects_missing_tables_and_invalid_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.toml"
            path.write_text("[experiment]\nname = 'bad'\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"\[data\]"):
                load_experiment_config(path)


class ExperimentRunnerTests(unittest.TestCase):
    def test_synthetic_artifacts_run_end_to_end_without_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "processed"
            data_dir.mkdir()
            write_synthetic_artifacts(data_dir)
            output_dir = root / "run"
            config = ExperimentConfig(
                name="synthetic_smoke",
                dataset="synthetic",
                processed_data_dir=data_dir,
                output_dir=output_dir,
                artifact_repository="team/synthetic",
                artifact_revision="frozen-revision",
                expected_protocol_version="synthetic_v1",
                expected_checksums=load_processed_dataset(data_dir).checksums,
                hidden_size=8,
                batch_size=2,
                learning_rate=0.01,
                max_epochs=2,
                patience=2,
                seed=17,
                device="cpu",
                deterministic=True,
            )

            results = run_experiment(config)

            self.assertTrue((output_dir / "best_model.pt").is_file())
            self.assertTrue((output_dir / "results.json").is_file())
            self.assertIn(results["best_epoch"], (1, 2))
            self.assertEqual(results["test"]["examples"], 2)
            self.assertEqual(results["filtering_protocol"], "synthetic_test")
            self.assertEqual(len(results["history"]), 2)
            saved = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["checkpoint_metric"], "validation_mrr@20")
            self.assertEqual(saved["dataset_checksums"], results["dataset_checksums"])
            self.assertEqual(
                saved["dataset_artifact"]["revision"], "frozen-revision"
            )

            with self.assertRaises(FileExistsError):
                run_experiment(config)

            bad_config = ExperimentConfig(
                name="wrong_artifact",
                dataset="synthetic",
                processed_data_dir=data_dir,
                output_dir=root / "bad-run",
                expected_protocol_version="different_protocol",
                device="cpu",
            )
            with self.assertRaisesRegex(ValueError, "protocol does not match"):
                run_experiment(bad_config)
            self.assertFalse((root / "bad-run").exists())


if __name__ == "__main__":
    unittest.main()
