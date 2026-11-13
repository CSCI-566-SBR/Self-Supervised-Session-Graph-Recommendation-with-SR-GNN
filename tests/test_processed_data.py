import json
import pickle
import tempfile
import unittest
from pathlib import Path

from recsys.data.processed import load_processed_dataset, load_processed_split


def write_synthetic_artifacts(root: Path) -> None:
    artifacts = {
        "train.txt": ([[1], [1, 2], [2, 3]], [2, 3, 4]),
        "validation.txt": ([[1, 2], [3]], [3, 4]),
        "test.txt": ([[2, 3], [4]], [4, 5]),
    }
    for filename, value in artifacts.items():
        with (root / filename).open("wb") as handle:
            pickle.dump(value, handle)
    with (root / "item_mapping.pkl").open("wb") as handle:
        pickle.dump({f"raw-{item}": item for item in range(1, 6)}, handle)
    (root / "preprocessing_summary.json").write_text(
        json.dumps(
            {
                "dataset": "synthetic",
                "filtering_protocol": "synthetic_test",
                "num_items": 5,
            }
        ),
        encoding="utf-8",
    )


class ProcessedDataTests(unittest.TestCase):
    def test_loads_shared_splits_mapping_summary_and_checksums(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_synthetic_artifacts(root)

            dataset = load_processed_dataset(root)

            self.assertEqual(dataset.num_items, 5)
            self.assertEqual(dataset.train[0], ((1,), 2))
            self.assertEqual(dataset.validation[-1], ((3,), 4))
            self.assertEqual(dataset.test[-1], ((4,), 5))
            self.assertEqual(dataset.summary["filtering_protocol"], "synthetic_test")
            self.assertEqual(
                set(dataset.checksums),
                {"train", "validation", "test", "item_mapping", "summary"},
            )
            self.assertTrue(all(len(digest) == 64 for digest in dataset.checksums.values()))

    def test_rejects_invalid_item_ids_and_mismatched_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bad_ids = root / "bad_ids.txt"
            with bad_ids.open("wb") as handle:
                pickle.dump(([[1, 0]], [2]), handle)
            with self.assertRaisesRegex(ValueError, "invalid or out-of-range"):
                load_processed_split(bad_ids, num_items=3)

            mismatched = root / "mismatched.txt"
            with mismatched.open("wb") as handle:
                pickle.dump(([[1]], [2, 3]), handle)
            with self.assertRaisesRegex(ValueError, "counts differ"):
                load_processed_split(mismatched, num_items=3)

    def test_rejects_noncontiguous_mapping_and_summary_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_synthetic_artifacts(root)
            with (root / "item_mapping.pkl").open("wb") as handle:
                pickle.dump({"a": 1, "b": 3}, handle)
            with self.assertRaisesRegex(ValueError, "contiguous"):
                load_processed_dataset(root)

            write_synthetic_artifacts(root)
            (root / "preprocessing_summary.json").write_text(
                json.dumps({"num_items": 99}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_processed_dataset(root)

    def test_requires_every_expected_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "missing processed artifacts"):
                load_processed_dataset(temporary)


if __name__ == "__main__":
    unittest.main()
