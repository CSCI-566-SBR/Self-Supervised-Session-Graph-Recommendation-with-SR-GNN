import tempfile
import unittest
from pathlib import Path

import pandas as pd

from recsys.data.preprocess import chronological_split, expand_prefixes, fixed_point_filter, preprocess
from recsys.data.processed import load_processed_dataset


class PreprocessingTests(unittest.TestCase):
    def test_prefix_expansion(self) -> None:
        self.assertEqual(expand_prefixes([[1, 2, 3]]), ([[1], [1, 2]], [2, 3]))

    def test_filter_reaches_fixed_point(self) -> None:
        frame = pd.DataFrame(
            {
                "session": ["a", "a", "b", "b", "c", "c"],
                "item": ["x", "rare", "x", "y", "y", "z"],
                "timestamp": pd.date_range("2026-01-01", periods=6, freq="h"),
            }
        )
        filtered, stats = fixed_point_filter(frame, min_item_frequency=2)
        self.assertTrue(filtered.empty)
        self.assertEqual(stats["sessions_dropped"], 3)

    def test_diginetica_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw.csv"
            raw.write_text(
                "sessionId;userId;itemId;timeframe;eventdate\n"
                "1;u1;10;0;2026-01-01\n"
                "1;u1;11;1000;2026-01-01\n"
                "2;u2;10;0;2026-01-02\n"
                "2;u2;11;1000;2026-01-02\n"
                "3;u3;10;0;2026-01-10\n"
                "3;u3;11;1000;2026-01-10\n"
                "4;u4;10;0;2026-01-20\n"
                "4;u4;11;1000;2026-01-20\n",
                encoding="utf-8",
            )
            output = root / "processed"
            stats = preprocess("diginetica", raw, output, min_item_frequency=1)
            dataset = load_processed_dataset(output)
            self.assertEqual(stats["num_items"], 2)
            self.assertGreater(len(dataset.train), 0)
            self.assertGreater(len(dataset.validation), 0)
            self.assertGreater(len(dataset.test), 0)
            self.assertTrue((output / "preprocessing.log").read_text(encoding="utf-8"))
            self.assertFalse((output / "all_train_seq.txt").exists())

    def test_chronological_train_validation_test_split(self) -> None:
        frame = pd.DataFrame(
            {
                "session": ["train","train","validation","validation","test","test",],
                "item": ["a","b","a","b","a","b",],
                "timestamp": pd.to_datetime(
                    [
                        "2026-01-01",
                        "2026-01-01 00:01:00",
                        "2026-01-10",
                        "2026-01-10 00:01:00",
                        "2026-01-20",
                        "2026-01-20 00:01:00",
                    ],
                    format="mixed",
                ),
            }
        )

        (train,validation,test,validation_start,test_start,) = chronological_split(frame,validation_days=7, test_days=7,)

        self.assertEqual(set(train["session"]), {"train"})
        self.assertEqual(set(validation["session"]),{"validation"},)
        self.assertEqual(set(test["session"]), {"test"})

        self.assertLess(validation_start, test_start)

if __name__ == "__main__":
    unittest.main()
