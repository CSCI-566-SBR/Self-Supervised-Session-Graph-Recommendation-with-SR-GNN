import json
import pickle
import random
import tempfile
import unittest
from pathlib import Path

from recsys.data.perturb import generate, perturb_sequence


class EvaluationPerturbationTests(unittest.TestCase):
    def test_noise_preserves_minimum_length_and_id_range(self) -> None:
        noisy, changed = perturb_sequence([1, 2, 3, 4], 1.0, 6, random.Random(2026))
        self.assertGreaterEqual(len(noisy), 2)
        self.assertTrue(all(1 <= item <= 6 for item in noisy))
        self.assertGreater(changed, 0)

    def test_length_one_prefix_is_preserved(self) -> None:
        self.assertEqual(perturb_sequence([3], 0.2, 8, random.Random(2)), ([3], 0))

    def test_generation_is_deterministic_and_target_invariant(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequences = [[1], [1, 2], [2, 3, 4], [1, 3, 4, 5]]
            targets = [2, 3, 5, 2]
            with (root / "test.txt").open("wb") as handle:
                pickle.dump((sequences, targets), handle)
            outputs = generate(root, ratios=(0.10,), seed=2026, vocab_size=5)
            first_bytes = outputs[0.10].read_bytes()
            with outputs[0.10].open("rb") as handle:
                noisy_sequences, noisy_targets = pickle.load(handle)
            self.assertEqual(noisy_targets, targets)
            self.assertEqual(noisy_sequences[0], sequences[0])
            generate(root, ratios=(0.10,), seed=2026, vocab_size=5)
            self.assertEqual(outputs[0.10].read_bytes(), first_bytes)
            report = json.loads((root / "perturbation_report.json").read_text())
            self.assertEqual(report["ratios"]["0.10"]["changed_clicks"], 1)


if __name__ == "__main__":
    unittest.main()
