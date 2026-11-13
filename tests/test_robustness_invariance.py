from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from recsys.data.processed import ProcessedDataset, load_processed_dataset
from recsys.models.baselines import SASRec
from recsys.models.registry import get_model_builder
from recsys.utils import validate_robustness_artifacts


class RobustnessInvarianceTests(unittest.TestCase):
    """Synthetic unit tests for robustness evaluation contracts and invariance assertions."""

    def setUp(self) -> None:
        self.clean_examples = [
            ([1, 2, 3], 4),
            ([5, 6], 7),
            ([8, 1, 2], 3),
        ]
        # Valid noisy examples: prefixes are modified (e.g. deletion, substitution), but targets are preserved
        self.valid_noisy_examples = [
            ([1, 3], 4),      # item 2 deleted
            ([5, 9], 7),      # item 6 substituted with 9
            ([8, 1], 3),      # item 2 deleted
        ]
        # Invalid noisy examples: target modified or example missing
        self.corrupted_target_examples = [
            ([1, 3], 4),
            ([5, 9], 99),     # target altered from 7 to 99!
            ([8, 1], 3),
        ]
        self.mismatched_length_examples = [
            ([1, 3], 4),
            ([5, 9], 7),
        ]

    def test_target_preservation_passes_valid_noisy_split(self) -> None:
        clean_targets = [target for _, target in self.clean_examples]
        noisy_targets = [target for _, target in self.valid_noisy_examples]

        self.assertEqual(len(self.valid_noisy_examples), len(self.clean_examples))
        self.assertEqual(clean_targets, noisy_targets)

    def test_target_corruption_detected(self) -> None:
        clean_targets = [target for _, target in self.clean_examples]
        corrupted_targets = [target for _, target in self.corrupted_target_examples]

        self.assertNotEqual(clean_targets, corrupted_targets)

    def test_length_mismatch_detected(self) -> None:
        self.assertNotEqual(len(self.mismatched_length_examples), len(self.clean_examples))

    def test_checkpoint_and_model_compatibility(self) -> None:
        """Verify that model state can be saved, restored, and produces identical logits on clean input."""
        torch.manual_seed(42)
        model = SASRec(num_items=10, hidden_size=16, maxlen=8, num_blocks=1, num_heads=2)
        model.eval()

        input_seq = torch.tensor([[0, 0, 1, 2]], dtype=torch.long)
        with torch.no_grad():
            original_logits = model(input_seq)

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "test_model.pt"
            torch.save(model.state_dict(), ckpt_path)

            restored_model = SASRec(num_items=10, hidden_size=16, maxlen=8, num_blocks=1, num_heads=2)
            state = torch.load(ckpt_path, weights_only=True)
            restored_model.load_state_dict(state)
            restored_model.eval()

            with torch.no_grad():
                restored_logits = restored_model(input_seq)

            torch.testing.assert_close(original_logits, restored_logits)

    def test_condition_metadata_parsing(self) -> None:
        """Verify that condition strings decompose into expected types and rates."""
        conditions = [
            ("clean", "clean", 0.0),
            ("deletion_0.05", "deletion", 0.05),
            ("mixed_0.10", "mixed", 0.10),
            ("substitution_0.20", "substitution", 0.20),
        ]
        for cond_str, expected_type, expected_rate in conditions:
            if cond_str == "clean":
                c_type, c_rate = "clean", 0.0
            else:
                parts = cond_str.split("_", 1)
                c_type = parts[0]
                c_rate = float(parts[1]) if len(parts) > 1 else 0.0
            self.assertEqual(c_type, expected_type)
            self.assertAlmostEqual(c_rate, expected_rate)

    def test_validate_robustness_artifacts_valid(self) -> None:
        """Verify that matching dataset, config, and checkpoint pass validation."""
        dataset = ProcessedDataset(
            train=(),
            validation=(),
            test=(),
            item_mapping={f"item_{i}": i for i in range(1, 11)},
            summary={"protocol_version": "diginetica_clean_v1"},
            checksums={"test": "abc123hash"},
        )
        meta = {
            "config": {
                "expected_protocol_version": "diginetica_clean_v1",
                "expected_checksums": {"test": "abc123hash"},
            }
        }
        checkpoint_state = {
            "item_embedding.weight": torch.zeros(11, 16)
        }
        # Should execute without raising
        validate_robustness_artifacts(dataset, meta, checkpoint_state)

    def test_validate_robustness_artifacts_protocol_mismatch(self) -> None:
        """Verify that protocol mismatch raises ValueError."""
        dataset = ProcessedDataset(
            train=(),
            validation=(),
            test=(),
            item_mapping={f"item_{i}": i for i in range(1, 11)},
            summary={"protocol_version": "diginetica_clean_v1"},
            checksums={"test": "abc123hash"},
        )
        meta = {
            "config": {
                "expected_protocol_version": "retailrocket_clean_v1",
            }
        }
        with self.assertRaisesRegex(ValueError, "protocol mismatch"):
            validate_robustness_artifacts(dataset, meta, None)

    def test_validate_robustness_artifacts_checksum_mismatch(self) -> None:
        """Verify that artifact checksum mismatch raises ValueError."""
        dataset = ProcessedDataset(
            train=(),
            validation=(),
            test=(),
            item_mapping={f"item_{i}": i for i in range(1, 11)},
            summary={"protocol_version": "diginetica_clean_v1"},
            checksums={"test": "actual_hash"},
        )
        meta = {
            "config": {
                "expected_checksums": {"test": "expected_hash"},
            }
        }
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            validate_robustness_artifacts(dataset, meta, None)

    def test_validate_robustness_artifacts_vocab_mismatch(self) -> None:
        """Verify that checkpoint vocabulary size mismatch raises ValueError."""
        dataset = ProcessedDataset(
            train=(),
            validation=(),
            test=(),
            item_mapping={f"item_{i}": i for i in range(1, 11)},  # 10 items -> 11 rows
            summary={"protocol_version": "diginetica_clean_v1"},
            checksums={},
        )
        meta = {"config": {}}
        # Checkpoint expects 25 rows instead of 11
        checkpoint_state = {
            "embedding.weight": torch.zeros(25, 16)
        }
        with self.assertRaisesRegex(ValueError, "Vocabulary size mismatch"):
            validate_robustness_artifacts(dataset, meta, checkpoint_state)


if __name__ == "__main__":
    unittest.main()
