import math
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from recsys.data.session_graph import collate_session_graphs
from recsys.models.baselines import SRGNN
from recsys.reproducibility import set_random_seed
from recsys.training import evaluate_ranking, train_epoch


class FixedScores(torch.nn.Module):
    def __init__(self, scores: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("scores", scores)

    def forward(self, _batch):
        return self.scores


class TrainingSmokeTests(unittest.TestCase):
    def test_forward_backward_and_evaluation_pipeline(self) -> None:
        examples = [
            ([1], 2),
            ([1, 2], 3),
            ([2, 3], 4),
            ([3, 4, 3], 5),
        ]
        batches = DataLoader(
            examples,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_session_graphs,
        )
        model = SRGNN(num_items=5, hidden_size=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        loss = train_epoch(model, batches, optimizer, device="cpu")
        results = evaluate_ranking(model, batches, device="cpu", k=20)

        self.assertTrue(math.isfinite(loss))
        self.assertEqual(results.examples, len(examples))
        self.assertGreaterEqual(results.recall_at_k, 0.0)
        self.assertLessEqual(results.recall_at_k, 1.0)
        self.assertGreaterEqual(results.mrr_at_k, 0.0)
        self.assertLessEqual(results.mrr_at_k, 1.0)

    def test_vectorized_metrics_match_hand_calculated_ranks(self) -> None:
        batch = collate_session_graphs(
            [([1], 1), ([1], 2), ([1], 3)]
        )
        # Target ranks are 1, 2, and 3. At K=2, the last example is a miss.
        scores = torch.tensor(
            [
                [4.0, 3.0, 2.0, 1.0],
                [4.0, 3.0, 2.0, 1.0],
                [4.0, 3.0, 2.0, 1.0],
            ]
        )
        results = evaluate_ranking(FixedScores(scores), [batch], "cpu", k=2)

        self.assertAlmostEqual(results.recall_at_k, 2 / 3)
        self.assertAlmostEqual(results.mrr_at_k, 0.5)

    def test_seed_repeats_python_numpy_and_torch_initialization(self) -> None:
        set_random_seed(23, deterministic=True)
        first_numpy = np.random.random(3)
        first_model = SRGNN(num_items=5, hidden_size=4)

        set_random_seed(23, deterministic=True)
        second_numpy = np.random.random(3)
        second_model = SRGNN(num_items=5, hidden_size=4)

        np.testing.assert_array_equal(first_numpy, second_numpy)
        for first, second in zip(first_model.parameters(), second_model.parameters()):
            torch.testing.assert_close(first, second)


if __name__ == "__main__":
    unittest.main()
