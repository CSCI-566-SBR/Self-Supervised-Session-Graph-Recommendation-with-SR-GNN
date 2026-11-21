import math
import unittest

import torch

from recsys.data.graph_augmentations import GraphView, Transition
from recsys.data.session_graph import collate_session_graphs
from recsys.models.baselines.srgnn import SRGNN
from recsys.models.srgnn_ssl import (
    SRGNNSSL,
    collate_graph_views,
    session_contrastive_loss,
)
from recsys.training.baseline import recommendation_loss


class SRGNNSSLTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.encoder = SRGNN(num_items=9, hidden_size=8)
        self.batch = collate_session_graphs([
            ([1, 2, 3, 4, 5, 6], 7),
            ([2, 3, 4, 5, 6, 7], 8),
        ])

    def test_wrapper_reuses_existing_encoder_parameters(self):
        model = SRGNNSSL(self.encoder)

        self.assertIs(model.encoder, self.encoder)
        self.assertEqual(
            {id(parameter) for parameter in model.parameters()},
            {id(parameter) for parameter in self.encoder.parameters()},
        )

    def test_inference_matches_existing_encoder(self):
        model = SRGNNSSL(self.encoder)
        model.eval()

        with torch.no_grad():
            actual = model(self.batch)
            expected = self.encoder(self.batch)

        torch.testing.assert_close(actual, expected)
        self.assertEqual(tuple(actual.shape), (2, 9))

    def test_zero_ssl_weight_matches_recommendation_loss(self):
        model = SRGNNSSL(self.encoder, ssl_weight=0.0)

        expected = recommendation_loss(
            self.encoder(self.batch),
            self.batch.targets,
        )
        actual = model.training_loss(self.batch)

        torch.testing.assert_close(actual, expected)

    def test_both_policies_support_backward_and_matched_counts(self):
        for mode in ("uniform_edge", "recency"):
            with self.subTest(mode=mode):
                encoder = SRGNN(num_items=9, hidden_size=8)
                model = SRGNNSSL(
                    encoder,
                    augmentation_mode=mode,
                    deletion_budget=0.20,
                )
                loss = model.training_loss(self.batch)
                loss.backward()

                self.assertTrue(torch.isfinite(loss).item())
                gradients = [
                    parameter.grad
                    for parameter in model.parameters()
                    if parameter.grad is not None
                ]
                self.assertTrue(gradients)
                self.assertTrue(all(
                    torch.isfinite(gradient).all().item()
                    for gradient in gradients
                ))

                # Two prefixes, two views each, five transitions each.
                metrics = model.last_training_metrics
                self.assertEqual(
                    metrics["augmentation_original_transitions"], 20
                )
                self.assertEqual(
                    metrics["augmentation_deleted_transitions"], 4
                )

    def test_uniform_node_supports_backward_and_metrics(self):
        encoder = SRGNN(num_items=9, hidden_size=8)
        model = SRGNNSSL(
            encoder,
            augmentation_mode="uniform_node",
            deletion_budget=0.20,
        )
        loss = model.training_loss(self.batch)
        loss.backward()

        self.assertTrue(torch.isfinite(loss).item())
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(
            torch.isfinite(gradient).all().item()
            for gradient in gradients
        ))

        metrics = model.last_training_metrics
        self.assertEqual(
            metrics["augmentation_original_transitions"], 20
        )
        self.assertGreaterEqual(metrics["augmentation_deleted_transitions"], 0)
        self.assertGreaterEqual(metrics["ssl_loss"], 0.0)

    def test_training_preserves_input_batch(self):
        model = SRGNNSSL(self.encoder)
        names = (
            "node_ids",
            "adjacency",
            "alias_indices",
            "sequence_mask",
            "targets",
        )
        original = {
            name: getattr(self.batch, name).clone()
            for name in names
        }

        model.training_loss(self.batch)

        for name in names:
            with self.subTest(field=name):
                self.assertTrue(torch.equal(
                    getattr(self.batch, name),
                    original[name],
                ))

    def test_view_batching_does_not_restore_deleted_edges(self):
        view = GraphView(
            sequence=(1, 2, 3),
            nodes=(1, 2, 3),
            edges=(Transition(source=2, target=3, index=1),),
        )
        batch = collate_graph_views([(view, 4)])

        # Incoming adjacency followed by outgoing adjacency.
        expected = torch.zeros((1, 3, 6))
        expected[0, 2, 1] = 1.0
        expected[0, 1, 5] = 1.0

        torch.testing.assert_close(batch.adjacency, expected)
        self.assertEqual(batch.targets.tolist(), [4])


class SessionContrastiveLossTests(unittest.TestCase):
    def test_matches_one_way_infonce_formula(self):
        embeddings = torch.eye(2)
        temperature = 0.5

        loss = session_contrastive_loss(
            embeddings,
            embeddings,
            temperature=temperature,
        )
        expected = math.log1p(math.exp(-1.0 / temperature))

        self.assertAlmostEqual(loss.item(), expected, places=6)

    def test_gradients_reach_both_views(self):
        first = torch.randn(3, 8, requires_grad=True)
        second = torch.randn(3, 8, requires_grad=True)

        session_contrastive_loss(first, second).backward()

        for view in (first, second):
            self.assertIsNotNone(view.grad)
            self.assertTrue(torch.isfinite(view.grad).all().item())
            self.assertGreater(view.grad.abs().sum().item(), 0.0)

    def test_single_example_has_zero_contrastive_loss(self):
        loss = session_contrastive_loss(
            torch.randn(1, 8),
            torch.randn(1, 8),
        )
        self.assertEqual(loss.item(), 0.0)

    def test_invalid_temperatures_are_rejected(self):
        embeddings = torch.eye(2)

        for temperature in (0.0, -0.1, float("nan"), float("inf")):
            with self.subTest(temperature=temperature):
                with self.assertRaises(ValueError):
                    session_contrastive_loss(
                        embeddings,
                        embeddings,
                        temperature=temperature,
                    )


if __name__ == "__main__":
    unittest.main()