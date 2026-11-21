import copy
import unittest
from unittest.mock import patch

import torch
import json
import pickle
import tempfile
from pathlib import Path

from recsys.data.session_graph import collate_session_graphs
from recsys.models.baselines.srgnn import SRGNN
from recsys.models.srgnn_ssl import SRGNNSSL
from recsys.training.baseline import (
    evaluate_ranking,
    recommendation_loss,
    train_epoch,
)
from recsys.data.processed import load_processed_dataset
from recsys.models.registry import get_model_builder
from recsys.training.config import ExperimentConfig
from recsys.training.runner import run_experiment


class SSLTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.encoder = SRGNN(num_items=9, hidden_size=8)
        self.batches = [
            collate_session_graphs([
                ([1, 2, 3, 4, 5, 6], 7),
                ([2, 3, 4, 5, 6, 7], 8),
            ]),
            collate_session_graphs([
                ([3, 4, 5, 6, 7, 8], 9),
            ]),
        ]

    def manual_baseline_epoch(self, model, optimizer):
        """Reproduce the original recommendation-only training steps."""
        model.train()
        total_loss = 0.0
        total_examples = 0

        for batch in self.batches:
            optimizer.zero_grad(set_to_none=True)
            loss = recommendation_loss(model(batch), batch.targets)
            loss.backward()
            optimizer.step()

            examples = batch.targets.shape[0]
            total_loss += loss.item() * examples
            total_examples += examples

        return total_loss / total_examples

    def assert_same_parameters(self, first, second):
        for actual, expected in zip(
            first.parameters(), second.parameters(), strict=True
        ):
            torch.testing.assert_close(actual, expected)

    def test_baseline_training_matches_original_steps(self):
        reference = copy.deepcopy(self.encoder)
        actual_optimizer = torch.optim.Adam(
            self.encoder.parameters(), lr=0.001
        )
        reference_optimizer = torch.optim.Adam(
            reference.parameters(), lr=0.001
        )

        expected_loss = self.manual_baseline_epoch(
            reference, reference_optimizer
        )
        actual_loss = train_epoch(
            self.encoder, self.batches, actual_optimizer, "cpu"
        )

        self.assertAlmostEqual(actual_loss, expected_loss, places=7)
        self.assert_same_parameters(self.encoder, reference)
        self.assertFalse(
            hasattr(self.encoder, "epoch_training_metrics")
        )

    def test_zero_ssl_weight_matches_baseline_updates(self):
        reference = copy.deepcopy(self.encoder)
        model = SRGNNSSL(self.encoder, ssl_weight=0.0)
        actual_optimizer = torch.optim.Adam(
            model.parameters(), lr=0.001
        )
        reference_optimizer = torch.optim.Adam(
            reference.parameters(), lr=0.001
        )

        expected_loss = self.manual_baseline_epoch(
            reference, reference_optimizer
        )
        actual_loss = train_epoch(
            model, self.batches, actual_optimizer, "cpu"
        )

        self.assertAlmostEqual(actual_loss, expected_loss, places=7)
        self.assert_same_parameters(model.encoder, reference)

    def test_ssl_epoch_records_joint_loss_and_deletion_counts(self):
        for mode in ("uniform_edge", "recency"):
            with self.subTest(mode=mode):
                model = SRGNNSSL(
                    copy.deepcopy(self.encoder),
                    augmentation_mode=mode,
                    deletion_budget=0.20,
                    ssl_weight=0.01,
                )
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=0.001
                )

                loss = train_epoch(
                    model, self.batches, optimizer, "cpu"
                )
                metrics = model.epoch_training_metrics

                self.assertEqual(
                    metrics["augmentation_original_transitions"], 30
                )
                self.assertEqual(
                    metrics["augmentation_deleted_transitions"], 6
                )
                self.assertAlmostEqual(
                    metrics["realized_deletion_rate"], 0.20
                )
                self.assertAlmostEqual(
                    loss,
                    metrics["recommendation_loss"]
                    + model.ssl_weight * metrics["ssl_loss"],
                    places=6,
                )

    def test_evaluation_does_not_call_ssl_training_loss(self):
        model = SRGNNSSL(self.encoder)

        with patch.object(
            model,
            "training_loss",
            side_effect=AssertionError(
                "Evaluation must not generate SSL training views"
            ),
        ):
            actual = evaluate_ranking(model, self.batches, "cpu")

        expected = evaluate_ranking(
            self.encoder, self.batches, "cpu"
        )

        self.assertEqual(actual.examples, expected.examples)
        self.assertEqual(actual.metrics, expected.metrics)
        self.assertAlmostEqual(
            actual.val_loss, expected.val_loss, places=7
        )

    def test_runner_saves_and_reloads_ssl_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "processed"
            data_dir.mkdir()

            prefixes = [
                [1, 2, 3, 4, 5, 6],
                [2, 3, 4, 5, 6, 7],
                [3, 4, 5, 6, 7, 8],
            ]
            targets = [7, 8, 9]
            splits = {
                "train.txt": (prefixes, targets),
                "validation.txt": (prefixes[:2], targets[:2]),
                "test.txt": (prefixes[:2], targets[:2]),
            }

            for filename, values in splits.items():
                with (data_dir / filename).open("wb") as handle:
                    pickle.dump(values, handle)

            with (data_dir / "item_mapping.pkl").open("wb") as handle:
                pickle.dump(
                    {f"raw-{item}": item for item in range(1, 10)},
                    handle,
                )

            (data_dir / "preprocessing_summary.json").write_text(
                json.dumps({
                    "dataset": "synthetic",
                    "filtering_protocol": "synthetic_test",
                    "protocol_version": "synthetic_v1",
                    "num_items": 9,
                }),
                encoding="utf-8",
            )
            dataset = load_processed_dataset(data_dir)

            for mode in ("uniform_edge", "recency"):
                with self.subTest(mode=mode):
                    output_dir = root / mode
                    config = ExperimentConfig(
                        name=f"ssl_smoke_{mode}",
                        dataset="synthetic",
                        processed_data_dir=data_dir,
                        output_dir=output_dir,
                        model_type="srgnn_ssl",
                        model_params={
                            "hidden_size": 8,
                            "gnn_steps": 1,
                            "hybrid": True,
                            "augmentation_mode": mode,
                            "deletion_budget": 0.20,
                            "ssl_weight": 0.01,
                            "temperature": 0.2,
                            "recency_gamma": 2.0,
                            "augmentation_seed": 2026,
                        },
                        expected_protocol_version="synthetic_v1",
                        expected_checksums=dataset.checksums,
                        batch_size=2,
                        max_epochs=1,
                        patience=1,
                        seed=2028,
                        device="cpu",
                        num_workers=0,
                    )

                    results = run_experiment(config)

                    checkpoint = output_dir / "best_model.pt"
                    self.assertTrue(checkpoint.is_file())
                    saved = json.loads(
                        (output_dir / "results.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    self.assertEqual(saved["model_type"], "srgnn_ssl")
                    self.assertEqual(saved["best_epoch"], 1)
                    self.assertEqual(saved["test"]["examples"], 2)
                    self.assertEqual(
                        saved["dataset_checksums"], dataset.checksums
                    )

                    metrics = saved["history"][0]["ssl"]
                    self.assertEqual(
                        metrics["augmentation_original_transitions"], 30
                    )
                    self.assertEqual(
                        metrics["augmentation_deleted_transitions"], 6
                    )
                    self.assertAlmostEqual(
                        metrics["realized_deletion_rate"], 0.20
                    )

                    builder = get_model_builder("srgnn_ssl")
                    restored, collate_fn = builder(
                        config.model_params, dataset
                    )
                    restored.load_state_dict(torch.load(
                        checkpoint,
                        map_location="cpu",
                        weights_only=True,
                    ))
                    test_batch = collate_fn(list(dataset.test))
                    evaluated = evaluate_ranking(
                        restored, [test_batch], "cpu"
                    )

                    for name, value in evaluated.metrics.items():
                        self.assertAlmostEqual(
                            value, results["test"][name], places=7
                        )
                    self.assertAlmostEqual(
                        evaluated.val_loss,
                        results["test"]["loss"],
                        places=6,
                    )
                    self.assertEqual(
                        load_processed_dataset(data_dir).checksums,
                        dataset.checksums,
                    )


if __name__ == "__main__":
    unittest.main()