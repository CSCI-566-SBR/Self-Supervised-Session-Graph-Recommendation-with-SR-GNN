import tempfile
import unittest
from pathlib import Path

import pandas as pd

from recsys.data.artifacts import load_pickle
from recsys.data.preprocess import expand_prefixes, fixed_point_filter, preprocess


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
                "1;u1;10;0;2026-01-01\n1;u1;11;1000;2026-01-01\n"
                "2;u2;10;0;2026-01-02\n2;u2;11;1000;2026-01-02\n"
                "3;u3;10;0;2026-01-20\n3;u3;11;1000;2026-01-20\n",
                encoding="utf-8",
            )
            output = root / "processed"
            stats = preprocess("diginetica", raw, output, min_item_frequency=1)
            train_sequences, train_targets = load_pickle(output / "train.txt")
            test_sequences, test_targets = load_pickle(output / "test.txt")
            self.assertEqual(stats["vocab_size"], 2)
            self.assertEqual(len(train_sequences), len(train_targets))
            self.assertEqual(len(test_sequences), len(test_targets))
            self.assertTrue((output / "preprocessing.log").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
