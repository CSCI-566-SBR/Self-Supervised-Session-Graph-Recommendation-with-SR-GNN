import tempfile
import unittest
from pathlib import Path

from recsys.data.artifacts import save_pickle
from recsys.data.validate import validate


class ArtifactValidationTests(unittest.TestCase):
    def test_valid_target_invariant_noise(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clean = ([[1], [1, 2], [2, 3, 4]], [2, 3, 1])
            save_pickle(clean, root / "test.txt")
            save_pickle({"a": 1, "b": 2, "c": 3, "d": 4}, root / "item_mapping.pkl")
            for ratio in (0.05, 0.10, 0.20):
                save_pickle(([[1], [4, 2], [2, 4]], clean[1]), root / f"test_noise_{ratio:.2f}.txt")
            report = validate(root)
            self.assertEqual(report["status"], "valid")
            self.assertTrue(all(entry["targets_identical"] for entry in report["noise"].values()))


if __name__ == "__main__":
    unittest.main()
