import math
import unittest

import torch

from recsys.data.graph_augmentations import GraphView, Transition
from recsys.data.session_graph import collate_session_graphs
from recsys.models.baselines.srgnn import SRGNN
from recsys.data.ssl_graph import SSLGraphCollator, collate_graph_views
from recsys.models.srgnn_ssl import (
    SRGNNSSL,
    session_contrastive_loss,
)
from recsys.training.baseline import recommendation_loss
from torch.utils.data import DataLoader



class SRGNNSSLTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.encoder = SRGNN(num_items=9, hidden_size=8)
        self.batch = collate_session_graphs([
            ([1, 2, 3, 4, 5, 6], 7),
            ([2, 3, 4, 5, 6, 7], 8),
        ])

    def make_training_batch(self, model):
        return model.make_training_collator()([
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
        actual, _ = model.training_loss(self.batch)

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
                batch = self.make_training_batch(model)
                loss, metrics = model.training_loss(batch)
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

                self.assertEqual(
                    metrics["augmentation_original_transitions"].total,
                    20,
                )
                self.assertEqual(
                    metrics["augmentation_deleted_transitions"].total,
                    4,
                )
                self.assertEqual(
                    metrics["augmentation_zero_deletion_prefixes"].total,
                    0,
                )

    def test_training_preserves_input_batch(self):
        model = SRGNNSSL(self.encoder)
        batch = self.make_training_batch(model)
        names = (
            "node_ids",
            "adjacency",
            "alias_indices",
            "sequence_mask",
            "targets",
        )
        original = {
            (view_name, field): getattr(
                getattr(batch, view_name), field
            ).clone()
            for view_name in ("clean", "first_view", "second_view")
            for field in names
        }

        model.training_loss(batch)

        for (view_name, field), expected in original.items():
            with self.subTest(view=view_name, field=field):
                actual = getattr(getattr(batch, view_name), field)
                self.assertTrue(torch.equal(actual, expected))

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

class SSLGraphCollationTests(unittest.TestCase):
    def setUp(self):
        self.examples = [
            ([1, 2, 3, 4, 5, 6], 7),
            ([2, 3, 4, 5, 6, 7], 8),
        ]

    def make_collator(self):
        return SSLGraphCollator(
            mode="recency",
            budget=0.20,
            gamma=2.0,
            seed=2026,
        )

    def assert_graph_equal(self, first, second):
        for name in (
            "node_ids",
            "adjacency",
            "alias_indices",
            "sequence_mask",
            "targets",
        ):
            torch.testing.assert_close(
                getattr(first, name),
                getattr(second, name),
            )

    def test_template_matches_standalone_collation(self):
        clean = collate_session_graphs([
            ([1, 2, 1, 2, 3], 4),
            ([3, 4], 5),
        ])
        examples = [
            (
                GraphView(
                    sequence=(1, 2, 1, 2, 3),
                    nodes=(1, 2, 3),
                    edges=(
                        Transition(1, 2, 0),
                        Transition(1, 2, 2),
                        Transition(2, 3, 3),
                    ),
                ),
                4,
            ),
            (
                GraphView(
                    sequence=(3, 4),
                    nodes=(3, 4),
                    edges=(Transition(3, 4, 0),),
                ),
                5,
            ),
        ]

        standalone = collate_graph_views(examples)
        reused = collate_graph_views(examples, template=clean)

        self.assert_graph_equal(reused, standalone)
        for name in (
            "node_ids",
            "alias_indices",
            "sequence_mask",
            "targets",
        ):
            self.assertIs(
                getattr(reused, name),
                getattr(clean, name),
            )

    def test_mismatched_template_is_rejected(self):
        clean = collate_session_graphs([self.examples[0]])
        different_view = GraphView(
            sequence=(1, 2),
            nodes=(1, 2),
            edges=(Transition(1, 2, 0),),
        )

        with self.assertRaisesRegex(
            ValueError, "unchanged item sequence"
        ):
            collate_graph_views(
                [(different_view, 7)],
                template=clean,
            )

    def test_collator_builds_cpu_views_and_preserves_targets(self):
        batch = self.make_collator()(self.examples)

        self.assertEqual(batch.targets.tolist(), [7, 8])
        self.assertEqual(batch.original_transitions, 20)
        self.assertEqual(batch.deleted_transitions, 4)
        self.assertEqual(batch.prefixes, 2)
        self.assertEqual(batch.zero_deletion_prefixes, 0)

        for view in (batch.clean, batch.first_view, batch.second_view):
            self.assertEqual(view.adjacency.device.type, "cpu")
            torch.testing.assert_close(view.targets, batch.targets)
            self.assertIs(view.node_ids, batch.clean.node_ids)
            self.assertIs(
                view.sequence_mask, batch.clean.sequence_mask
            )

        moved = batch.to("cpu")
        self.assert_graph_equal(moved.clean, batch.clean)
        self.assert_graph_equal(moved.first_view, batch.first_view)
        self.assert_graph_equal(moved.second_view, batch.second_view)
        self.assertIs(
            moved.first_view.node_ids, moved.clean.node_ids
        )

    def test_worker_loading_is_reproducible(self):
        runs = []

        for _ in range(2):
            loader = DataLoader(
                self.examples * 2,
                batch_size=2,
                shuffle=False,
                num_workers=2,
                collate_fn=self.make_collator(),
                generator=torch.Generator().manual_seed(2028),
            )
            runs.append(list(loader))

        self.assertEqual(len(runs[0]), 2)
        self.assertEqual(len(runs[1]), 2)

        for first, second in zip(runs[0], runs[1], strict=True):
            self.assert_graph_equal(first.clean, second.clean)
            self.assert_graph_equal(
                first.first_view, second.first_view
            )
            self.assert_graph_equal(
                first.second_view, second.second_view
            )
            self.assertEqual(first.deleted_transitions, 4)
            self.assertEqual(second.deleted_transitions, 4)

if __name__ == "__main__":
    unittest.main()