"""Small integration checks, never reported as real recommendation results."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from model2_io import read_json, sha
from test_model2_finish_io import fixture
from recsys.model2.runner import train_run
from recsys.model2.finish import evaluate_full


def args(data, output, epochs, resume=False):
    return SimpleNamespace(data=data, output=output, epochs=epochs, resume=resume,
        patience=10, batch_size=2, hidden_size=6, cpu_threads=1, debug_limit=0,
        seed=2028, kind="deletion", ratio=.1, fraction=.5, noise_seed=2026,
        deterministic=True, device="cpu", num_workers=0, fused=False)


class FinishNeuralTests(unittest.TestCase):
    def test_train_checkpoint_and_full_frozen_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = fixture(root / "data")
            run = root / "run"
            train_run(args(root / "data", run, 2), baseline_config=config)
            before = sha(run / "best_model.pt")
            values = evaluate_full(run, root / "data", "toy", config, "cpu")
            self.assertEqual(len(values), 10)
            self.assertEqual(sha(run / "best_model.pt"), before)
            for k in (10, 20):
                for m in ("recall", "mrr", "ndcg"):
                    self.assertTrue(0 <= values["clean"][f"{m}@{k}"] <= 1)
            self.assertEqual(read_json(run / "results.json")["training_kind"], "fixed_deletion")
            # Completed evaluation is reused, without another training update.
            self.assertEqual(evaluate_full(run, root / "data", "toy", config, "cpu"), values)

    def test_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = fixture(root / "data")
            train_run(args(root / "data", root / "full", 2), baseline_config=config)
            train_run(args(root / "data", root / "resumed", 1), baseline_config=config)
            train_run(args(root / "data", root / "resumed", 2, resume=True), baseline_config=config)
            x = torch.load(root / "full/last_training.pt", weights_only=True)
            y = torch.load(root / "resumed/last_training.pt", weights_only=True)
            for key in x["model"]:
                torch.testing.assert_close(x["model"][key], y["model"][key], rtol=0, atol=0)


if __name__ == "__main__": unittest.main()
