import math
import random
import unittest
from unittest.mock import patch

import torch

from recsys.data.graph_augmentations import GraphView, Transition, make_two_views
from recsys.data.session_graph import collate_session_graphs
from recsys.models.baselines.srgnn import SRGNN
from recsys.data.ssl_graph import SSLGraphCollator, collate_graph_views
from recsys.models.srgnn_ssl import (
    SRGNNSSL,
    session_contrastive_loss,
)
from recsys.training.baseline import recommendation_loss
from torch.nn import functional as F
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
    def test_mask_filters_both_axes_scales_and_preserves_positive_alignment(self):
        torch.manual_seed(42)
        first = torch.randn(5, 8, dtype=torch.float64, requires_grad=True)
        second = torch.randn(5, 8, dtype=torch.float64, requires_grad=True)
        mask = torch.tensor([True, False, True, False, True])
        actual = session_contrastive_loss(first, second, pair_mask=mask)
        logits = F.normalize(first[mask], dim=1) @ F.normalize(second[mask], dim=1).T / 0.2
        expected = F.cross_entropy(logits, torch.arange(3)) * (3 / 5)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        for view in (first, second):
            self.assertEqual(view.grad[~mask].count_nonzero().item(), 0)
            self.assertGreater(view.grad[mask].abs().sum().item(), 0)
        # Altering excluded negative candidates must have no effect either.
        changed = second.detach().clone()
        changed[~mask] = 1000
        torch.testing.assert_close(
            session_contrastive_loss(first.detach(), changed, pair_mask=mask), actual.detach()
        )

    def test_mask_with_zero_or_one_pair_returns_differentiable_zero(self):
        for values in ([False, False, False], [False, True, False]):
            first = torch.randn(3, 8, requires_grad=True)
            second = torch.randn(3, 8, requires_grad=True)
            loss = session_contrastive_loss(first, second, pair_mask=torch.tensor(values))
            self.assertEqual(loss.item(), 0)
            loss.backward()
            self.assertEqual(first.grad.count_nonzero().item(), 0)
            self.assertEqual(second.grad.count_nonzero().item(), 0)

    def test_all_true_mask_matches_original_loss(self):
        first, second = torch.randn(4, 8), torch.randn(4, 8)
        torch.testing.assert_close(
            session_contrastive_loss(first, second),
            session_contrastive_loss(first, second, pair_mask=torch.ones(4, dtype=torch.bool)),
        )

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
    def test_repeated_edge_deletions_do_not_count_as_distinct_graphs(self):
        batch = self.make_collator()([([1, 2, 1, 2, 1, 2], 3)])
        self.assertEqual(batch.deleted_transitions, 2)
        self.assertEqual(batch.distinct_pairs, 0)
        self.assertEqual(batch.distinct_pair_mask.tolist(), [False])
        torch.testing.assert_close(batch.first_view.adjacency, batch.clean.adjacency)
        torch.testing.assert_close(batch.second_view.adjacency, batch.clean.adjacency)

    def test_same_modified_graphs_are_excluded_even_when_both_differ_from_clean(self):
        view = GraphView((1, 2, 3), (1, 2, 3), (Transition(2, 3, 1),))
        with patch("recsys.data.ssl_graph.make_two_views", return_value=(view, view)):
            batch = self.make_collator()([([1, 2, 3], 4)])
        self.assertFalse(torch.equal(batch.first_view.adjacency, batch.clean.adjacency))
        self.assertEqual(batch.distinct_pair_mask.tolist(), [False])

    def test_substitution_rebuilds_and_transfers_independent_node_metadata(self):
        def view(sequence):
            return GraphView(tuple(sequence), tuple(dict.fromkeys(sequence)), tuple(
                Transition(a, b, i) for i, (a, b) in enumerate(zip(sequence, sequence[1:]))
            ))
        prefix = [1, 2, 1, 2]
        first, second = view([4, 2, 1, 2]), view([1, 3, 1, 2])
        collator = SSLGraphCollator(
            mode="recency_substitution", budget=0.2, gamma=2.0, seed=2026, num_items=5
        )
        with patch("recsys.data.ssl_graph.make_two_views", return_value=(first, second)):
            batch = collator([(prefix, 5)])
        self.assertEqual(batch.distinct_pair_mask.tolist(), [True])
        self.assertEqual(batch.substituted_items, 2)
        self.assertFalse(torch.equal(batch.first_view.node_ids, batch.clean.node_ids))
        self.assertFalse(torch.equal(batch.first_view.alias_indices, batch.clean.alias_indices))
        for actual, sequence in ((batch.first_view, first.sequence), (batch.second_view, second.sequence)):
            expected = collate_session_graphs([(sequence, 5)])
            self.assert_graph_equal(actual, expected)
            recovered = actual.node_ids[0].gather(0, actual.alias_indices[0])
            self.assertEqual(recovered[actual.sequence_mask[0]].tolist(), list(sequence))
            self.assertEqual(recovered[-1].item(), prefix[-1])
        for moved in (batch.to("cpu"), batch._map_tensors(lambda tensor: tensor.clone())):
            self.assert_graph_equal(moved.first_view, batch.first_view)
            self.assert_graph_equal(moved.second_view, batch.second_view)
            self.assertEqual(moved.distinct_pair_mask.tolist(), [True])

    def assert_matches_unpadded_graph(self, actual, expected):
        nodes = expected.node_ids.shape[1]
        padded_nodes = actual.node_ids.shape[1]
        torch.testing.assert_close(actual.node_ids[:, :nodes], expected.node_ids)
        self.assertEqual(actual.node_ids[:, nodes:].count_nonzero().item(), 0)
        for name in ("alias_indices", "sequence_mask", "targets"):
            torch.testing.assert_close(getattr(actual, name), getattr(expected, name))
        adjacency = torch.cat((
            actual.adjacency[:, :nodes, :nodes],
            actual.adjacency[:, :nodes, padded_nodes:padded_nodes + nodes],
        ), dim=2)
        torch.testing.assert_close(adjacency, expected.adjacency, rtol=0, atol=0)
        self.assertEqual(actual.adjacency[:, nodes:, :].count_nonzero().item(), 0)
        self.assertEqual(actual.adjacency[:, :, nodes:padded_nodes].count_nonzero().item(), 0)
        self.assertEqual(actual.adjacency[:, :, padded_nodes + nodes:].count_nonzero().item(), 0)

    def test_substitution_common_padding_reuses_fused_encoding_and_matches_reference(self):
        def view(sequence):
            return GraphView(tuple(sequence), tuple(dict.fromkeys(sequence)), tuple(
                Transition(a, b, i) for i, (a, b) in enumerate(zip(sequence, sequence[1:]))
            ))

        examples = [([1], 9), ([2, 2], 9), ([1, 2, 1, 2], 9), ([1, 2, 1, 3], 9)]
        sequences = [
            ([1], [1]),
            ([2, 2], [2, 2]),
            ([2, 2, 1, 2], [4, 2, 1, 2]),
            ([1, 1, 1, 3], [4, 2, 1, 3]),
        ]
        pairs = [(view(first), view(second)) for first, second in sequences]
        collator = SSLGraphCollator(
            mode="recency_substitution", budget=0.2, gamma=2.0, seed=2026, num_items=9
        )
        # The optimized path should build all three batches without invoking
        # either of the standalone, per-session tensor collators.
        with (
            patch("recsys.data.ssl_graph.make_two_views", side_effect=pairs),
            patch("recsys.data.ssl_graph.collate_session_graphs", side_effect=AssertionError),
            patch("recsys.data.ssl_graph.collate_graph_views", side_effect=AssertionError),
        ):
            batch = collator(examples)

        references = [collate_session_graphs(examples)] + [
            collate_graph_views([(pair[index], target) for pair, (_, target) in zip(pairs, examples)])
            for index in range(2)
        ]
        self.assertEqual([graph.node_ids.shape[1] for graph in references], [3, 2, 4])
        for actual, expected in zip((batch.clean, batch.first_view, batch.second_view), references):
            self.assertEqual(actual.node_ids.shape[1], 4)
            self.assert_matches_unpadded_graph(actual, expected)
            recovered = actual.node_ids.gather(1, actual.alias_indices)
            lengths = actual.sequence_mask.sum(dim=1)
            self.assertEqual(recovered[torch.arange(4), lengths - 1].tolist(), [1, 2, 2, 3])
        self.assertEqual(batch.distinct_pair_mask.tolist(), [False, False, True, True])
        self.assertEqual(batch.substituted_items, 4)

        encoder = SRGNN(num_items=9, hidden_size=8)
        model = SRGNNSSL(
            encoder, augmentation_mode="recency_substitution",
            deletion_budget=0.2, mask_identical_views=True,
        )
        reference_embeddings = [encoder.encode(graph) for graph in references]
        reference_logits = reference_embeddings[0] @ encoder.item_embedding.weight[1:].T
        reference_loss = recommendation_loss(reference_logits, references[0].targets)
        reference_loss = reference_loss + model.ssl_weight * session_contrastive_loss(
            reference_embeddings[1], reference_embeddings[2], pair_mask=batch.distinct_pair_mask
        )
        reference_loss.backward()
        gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        with patch.object(encoder, "encode", wraps=encoder.encode) as encode:
            actual_loss, _ = model.training_loss(batch)
        self.assertEqual(encode.call_count, 1)
        self.assertEqual(encode.call_args.args[0].node_ids.shape, (12, 4))
        torch.testing.assert_close(actual_loss, reference_loss, rtol=1e-5, atol=1e-6)
        actual_loss.backward()
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter.grad, gradients[name], rtol=1e-4, atol=1e-6)

    def test_optimized_substitution_preserves_view_rng_streams(self):
        examples = [([1], 9), ([1, 2], 9), ([1, 2, 1, 2, 3, 4], 9)]
        for rounding in ("nearest", "stochastic"):
            collator = SSLGraphCollator(
                mode="recency_substitution", budget=0.2, gamma=2.0, seed=2026,
                num_items=9, budget_rounding=rounding,
            )
            rng, budget_rng = random.Random(2026), random.Random("2026:budget")
            for _ in range(2):
                pairs = [make_two_views(
                    prefix, mode="recency_substitution", budget=0.2, gamma=2.0,
                    rng=rng, budget_rng=budget_rng, budget_rounding=rounding,
                    num_items=9, target=target,
                ) for prefix, target in examples]
                batch = collator(examples)
                for index, actual in enumerate((batch.first_view, batch.second_view)):
                    expected = collate_graph_views([
                        (pair[index], target) for pair, (_, target) in zip(pairs, examples)
                    ])
                    self.assert_matches_unpadded_graph(actual, expected)
                self.assertEqual(collator._rng.getstate(), rng.getstate())
                self.assertEqual(collator._budget_rng.getstate(), budget_rng.getstate())

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

    def deletion_counts(self, batch):
        # These fixtures have no repeated transitions, so nonzero
        # outgoing edges equal retained transition occurrences.
        max_nodes = batch.clean.node_ids.shape[1]
        original = batch.clean.sequence_mask.sum(dim=1) - 1
        first_retained = torch.count_nonzero(
            batch.first_view.adjacency[:, :, max_nodes:],
            dim=(1, 2),
        )
        second_retained = torch.count_nonzero(
            batch.second_view.adjacency[:, :, max_nodes:],
            dim=(1, 2),
        )

        torch.testing.assert_close(first_retained, second_retained)
        counts = (original - first_retained).tolist()
        self.assertEqual(sum(counts) * 2, batch.deleted_transitions)
        self.assertEqual(
            counts.count(0), batch.zero_deletion_prefixes
        )
        return counts

    def test_explicit_nearest_matches_default_collator(self):
        default = self.make_collator()
        explicit = SSLGraphCollator(
            mode="recency",
            budget=0.20,
            gamma=2.0,
            seed=2026,
            budget_rounding="nearest",
        )

        for _ in range(3):
            first = default(self.examples)
            second = explicit(self.examples)
            for name in ("clean", "first_view", "second_view"):
                self.assert_graph_equal(
                    getattr(first, name),
                    getattr(second, name),
                )

    def test_stochastic_worker_counts_match_between_policies(self):
        examples = [
            (list(range(1, length + 1)), length + 1)
            for length in (1, 2, 3, 6, 8, 11)
        ]
        runs = []

        # Repeat uniform to check reproducibility, then compare recency.
        for mode in ("uniform_edge", "uniform_edge", "recency"):
            loader = DataLoader(
                examples,
                batch_size=3,
                shuffle=False,
                num_workers=2,
                collate_fn=SSLGraphCollator(
                    mode=mode,
                    budget=0.10,
                    gamma=2.0,
                    seed=2026,
                    budget_rounding="stochastic",
                ),
                generator=torch.Generator().manual_seed(2028),
            )
            runs.append(list(loader))

        for batches in runs:
            self.assertEqual(len(batches), 2)

        for first, repeated, recency in zip(*runs, strict=True):
            uniform_counts = self.deletion_counts(first)
            self.assertEqual(
                uniform_counts, self.deletion_counts(repeated)
            )
            self.assertEqual(
                uniform_counts, self.deletion_counts(recency)
            )

            for name in ("clean", "first_view", "second_view"):
                self.assert_graph_equal(
                    getattr(first, name),
                    getattr(repeated, name),
                )

if __name__ == "__main__":
    unittest.main()
