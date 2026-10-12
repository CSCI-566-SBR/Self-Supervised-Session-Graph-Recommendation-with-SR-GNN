import unittest

import torch

from recsys.data.session_graph import collate_session_graphs
from recsys.models.baselines import SRGNN
from recsys.training import recommendation_loss


class SRGNNTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = SRGNN(num_items=9, hidden_size=8, gnn_steps=1)

    def test_forward_shape_and_finite_values(self) -> None:
        batch = collate_session_graphs([([1, 2, 1], 3), ([4, 5], 6)])
        logits = self.model(batch)

        self.assertEqual(tuple(logits.shape), (2, 9))
        self.assertTrue(torch.isfinite(logits).all())

    def test_batch_padding_does_not_change_valid_example(self) -> None:
        self.model.eval()
        short = collate_session_graphs([([1, 2], 3)])
        mixed = collate_session_graphs([([1, 2], 3), ([4, 5, 6, 7], 8)])

        with torch.no_grad():
            short_logits = self.model(short)[0]
            mixed_logits = self.model(mixed)[0]
        torch.testing.assert_close(short_logits, mixed_logits)

    def test_loss_is_finite_and_backward_reaches_model(self) -> None:
        batch = collate_session_graphs([([1, 2, 1], 3), ([4, 5], 6)])
        loss = recommendation_loss(self.model(batch), batch.targets)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        parameters_with_gradients = [
            parameter for parameter in self.model.parameters() if parameter.grad is not None
        ]
        self.assertTrue(parameters_with_gradients)
        self.assertTrue(
            all(torch.isfinite(parameter.grad).all() for parameter in parameters_with_gradients)
        )

    def test_invalid_model_and_target_arguments(self) -> None:
        with self.assertRaises(ValueError):
            SRGNN(num_items=0)
        with self.assertRaises(ValueError):
            SRGNN(num_items=3, hidden_size=0)

        logits = torch.zeros((1, 3))
        with self.assertRaises(ValueError):
            recommendation_loss(logits, torch.tensor([0]))
        with self.assertRaises(ValueError):
            recommendation_loss(logits, torch.tensor([4]))


if __name__ == "__main__":
    unittest.main()
