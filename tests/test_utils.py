from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import torch

from recsys.utils import (
    format_results_dict,
    get_git_commit,
    load_pickle,
    load_toml,
    make_data_loader,
    prepare_output_dir,
    resolve_device,
    resolve_num_workers,
    save_pickle,
    set_random_seed,
    sha256,
    sha256_file,
)


class UtilsTests(unittest.TestCase):
    def test_sha256_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            f = Path(tmp_dir) / "test.txt"
            f.write_text("hello world", encoding="utf-8")
            # sha256 of "hello world" is b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9
            expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
            self.assertEqual(sha256_file(f), expected)
            self.assertEqual(sha256(f), expected)

    def test_reproducibility(self) -> None:
        with self.assertRaises(ValueError):
            set_random_seed(-1)
        with self.assertRaises(ValueError):
            set_random_seed(True)  # type: ignore[arg-type]

        set_random_seed(42, deterministic=True)
        r1 = torch.randn(5)
        set_random_seed(42, deterministic=True)
        r2 = torch.randn(5)
        torch.testing.assert_close(r1, r2)

    def test_resolve_device(self) -> None:
        cpu_dev = resolve_device("cpu")
        self.assertEqual(cpu_dev.type, "cpu")
        auto_dev = resolve_device("auto")
        self.assertIn(auto_dev.type, ("cuda", "mps", "cpu"))

    def test_resolve_num_workers(self) -> None:
        cpu_dev = torch.device("cpu")
        self.assertEqual(resolve_num_workers(cpu_dev), 0)
        self.assertEqual(resolve_num_workers(cpu_dev, requested=3), 3)

        cuda_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if cuda_dev.type == "cuda":
            self.assertGreaterEqual(resolve_num_workers(cuda_dev), 1)

        os.environ["RECSYS_NUM_WORKERS"] = "5"
        try:
            self.assertEqual(resolve_num_workers(cpu_dev), 5)
        finally:
            del os.environ["RECSYS_NUM_WORKERS"]

    def test_pickle_io(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sub" / "data.pkl"
            payload = {"a": [1, 2, 3], "b": "test"}
            save_pickle(payload, path)
            restored = load_pickle(path)
            self.assertEqual(restored, payload)
            # Ensure atomic temp files are cleaned up
            temp_files = list(path.parent.glob(".tmp_*"))
            self.assertEqual(len(temp_files), 0)

    def test_toml_io(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "config.toml"
            path.write_text("[section]\nkey = 'value'\ncount = 10\nflag = true\n", encoding="utf-8")
            data = load_toml(path)
            self.assertEqual(data["section"]["key"], "value")
            self.assertEqual(data["section"]["count"], 10)
            self.assertTrue(data["section"]["flag"])

    def test_prepare_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out = Path(tmp_dir) / "new_dir"
            prepare_output_dir(out)
            self.assertTrue(out.is_dir())
            with self.assertRaises(FileExistsError):
                prepare_output_dir(out)

    def test_make_data_loader(self) -> None:
        data = [([1, 2], 3), ([4, 5], 6)]
        loader = make_data_loader(
            examples=data,
            batch_size=2,
            shuffle=False,
            seed=42,
            collate_fn=lambda b: b,
            num_workers=0,
        )
        batch = next(iter(loader))
        self.assertEqual(len(batch), 2)

    def test_get_git_commit(self) -> None:
        commit = get_git_commit()
        if commit is not None:
            self.assertEqual(len(commit), 40)


if __name__ == "__main__":
    unittest.main()
