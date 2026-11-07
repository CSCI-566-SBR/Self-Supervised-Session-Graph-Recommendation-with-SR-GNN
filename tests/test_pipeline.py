import tempfile
import unittest
from pathlib import Path

from recsys.data.pipeline import run_pipeline
from recsys.data.processed import load_processed_dataset


class PipelineTests(unittest.TestCase):
    def test_one_command_writes_clean_and_noisy_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw.csv"
            raw.write_text(
                "sessionId;userId;itemId;timeframe;eventdate\n"
                "1;u1;10;0;2026-01-01\n1;u1;11;1000;2026-01-01\n"
                "2;u2;10;0;2026-01-02\n2;u2;11;1000;2026-01-02\n"
                "3;u3;10;0;2026-01-10\n3;u3;11;1000;2026-01-10\n"
                "4;u4;10;0;2026-01-20\n4;u4;11;1000;2026-01-20\n",
                encoding="utf-8",
            )
            output = root / "output"
            manifest = run_pipeline(
                "diginetica",
                raw,
                output,
                min_item_frequency=1,
                ratios=(0.20,),
                corruption_types=("mixed",),
            )
            clean = load_processed_dataset(output / "clean")
            noisy = load_processed_dataset(output / "noisy" / "mixed_0.20")
            self.assertEqual(clean.train, noisy.train)
            self.assertEqual([target for _, target in clean.test], [target for _, target in noisy.test])
            self.assertIn("mixed_0.20", manifest["noisy"])
            self.assertTrue((output / "pipeline_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
