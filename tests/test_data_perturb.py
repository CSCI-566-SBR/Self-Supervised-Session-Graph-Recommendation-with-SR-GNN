import json
import random
import tempfile
import unittest
from pathlib import Path

from recsys.data.artifacts import save_pickle
from recsys.data.perturb import generate_noisy_datasets, perturb_sequence
from recsys.data.processed import load_processed_dataset


def write_clean_dataset(root: Path) -> None:
    save_pickle(([[1], [1, 2], [2, 3, 4], [1, 3, 4, 5]], [2, 3, 5, 2]), root / "train.txt")
    save_pickle(([[1], [2, 3]], [3, 4]), root / "validation.txt")
    save_pickle(([[1], [1, 2], [2, 3, 4], [1, 3, 4, 5]], [2, 3, 5, 2]), root / "test.txt")
    save_pickle({str(item): item for item in range(1, 6)}, root / "item_mapping.pkl")
    (root / "preprocessing_summary.json").write_text(
        json.dumps({"dataset": "synthetic", "num_items": 5}), encoding="utf-8"
    )


class EvaluationPerturbationTests(unittest.TestCase):
    def test_noise_preserves_minimum_length_and_id_range(self) -> None:
        noisy, changed = perturb_sequence(
            [1, 2, 3, 4], 1.0, 6, random.Random(2026), target=5
        )
        self.assertGreaterEqual(len(noisy), 2)
        self.assertTrue(all(1 <= item <= 6 for item in noisy))
        self.assertGreater(changed, 0)

    def test_length_one_prefix_is_preserved(self) -> None:
        self.assertEqual(perturb_sequence([3], 0.2, 8, random.Random(2)), ([3], 0))

    def test_generation_is_separate_deterministic_and_target_invariant(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clean_dir = root / "clean"
            write_clean_dataset(clean_dir)
            manifest = generate_noisy_datasets(
                clean_dir,
                root / "noisy",
                ratios=(0.50,),
                corruption_types=("mixed",),
            )
            destination = root / "noisy" / "mixed_0.50"
            first_bytes = (destination / "test.txt").read_bytes()
            clean = load_processed_dataset(clean_dir)
            noisy = load_processed_dataset(destination)
            self.assertEqual(
                [target for _, target in noisy.test], [target for _, target in clean.test]
            )
            self.assertEqual(noisy.train, clean.train)
            self.assertEqual(noisy.validation, clean.validation)
            self.assertEqual(noisy.test[0], clean.test[0])
            generate_noisy_datasets(
                clean_dir,
                root / "noisy",
                ratios=(0.50,),
                corruption_types=("mixed",),
            )
            self.assertEqual((destination / "test.txt").read_bytes(), first_bytes)
            self.assertIn("mixed_0.50", manifest)


if __name__ == "__main__":
    unittest.main()
