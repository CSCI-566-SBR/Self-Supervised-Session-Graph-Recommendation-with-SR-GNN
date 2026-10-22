import json
import tempfile
import unittest
from pathlib import Path

from recsys.data.artifacts import save_pickle
from recsys.data.perturb import generate_noisy_datasets
from recsys.data.validate import validate_pipeline


def write_clean_dataset(root: Path) -> None:
    value = ([[1], [1, 2], [2, 3, 4]], [2, 3, 1])
    save_pickle(value, root / "train.txt")
    save_pickle(value, root / "validation.txt")
    save_pickle(value, root / "test.txt")
    save_pickle({"a": 1, "b": 2, "c": 3, "d": 4}, root / "item_mapping.pkl")
    (root / "preprocessing_summary.json").write_text(
        json.dumps({"dataset": "synthetic", "num_items": 4}), encoding="utf-8"
    )


class ArtifactValidationTests(unittest.TestCase):
    def test_valid_target_invariant_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_clean_dataset(root / "clean")
            generate_noisy_datasets(
                root / "clean",
                root / "noisy",
                ratios=(0.20,),
                corruption_types=("deletion", "substitution", "mixed"),
            )
            report = validate_pipeline(root)
            self.assertEqual(report["status"], "valid")
            self.assertEqual(len(report["noisy"]), 3)


if __name__ == "__main__":
    unittest.main()
