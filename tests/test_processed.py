import json
import pickle
import tempfile
import unittest
from pathlib import Path

from recsys.data.artifacts import save_pickle
from recsys.data.processed import REQUIRED_FILES, load_processed_dataset


def write_dataset(root: Path) -> None:
    split = ([[1], [1, 2], [2, 4, 2]], [2, 3, 5])
    for filename in ("train.txt", "validation.txt", "test.txt"):
        save_pickle(split, root / filename)
    save_pickle({101: 1, 205: 2, 330: 3, 444: 4, 900: 5}, root / "item_mapping.pkl")
    (root / "preprocessing_summary.json").write_text(
        json.dumps({"dataset": "synthetic", "num_items": 5}), encoding="utf-8"
    )


class ProcessedDatasetTests(unittest.TestCase):
    def test_loads_and_normalizes_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_dataset(root)
            dataset = load_processed_dataset(root)
            self.assertEqual(dataset.num_items, 5)
            self.assertEqual(dataset.train[0], ((1,), 2))
            self.assertEqual(set(dataset.checksums), set(REQUIRED_FILES))

    def test_rejects_boolean_and_numpy_like_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_dataset(root)
            with (root / "train.txt").open("wb") as handle:
                pickle.dump(([[True]], [2]), handle)
            with self.assertRaisesRegex(ValueError, "Python int"):
                load_processed_dataset(root)


if __name__ == "__main__":
    unittest.main()
