import random
import unittest
from recsys.data.graph_augmentations import augment_recency_substitution
from unittest.mock import patch

from recsys.data.graph_augmentations import (
    augment_budgeted_edge_view,
    augment_session,
    make_two_views,
)
from recsys.data.graph_augmentations import (
    augment_budgeted_edge_view,
    augment_session,
    count_edge_deletions,
    make_two_views,
)

class BudgetedSSLTests(unittest.TestCase):
    MODES = ("uniform_edge", "recency")

    def make_view(self, prefix, mode, budget, seed=2026):
        return augment_budgeted_edge_view(
            prefix,
            mode=mode,
            budget=budget,
            rng=random.Random(seed),
            gamma=2.0,
        )

    def test_both_policies_remove_exact_same_count(self):
        prefix = list(range(1, 12))  # Ten transition occurrences.

        for mode in self.MODES:
            for budget, expected_deleted in ((0.10, 1), (0.20, 2)):
                with self.subTest(mode=mode, budget=budget):
                    view = self.make_view(prefix, mode, budget)
                    self.assertEqual(10 - len(view.edges), expected_deleted)
                    self.assertEqual(view.sequence, tuple(prefix))

    def test_final_transition_is_protected(self):
        for mode in self.MODES:
            with self.subTest(mode=mode):
                view = self.make_view([1, 2, 3, 4, 5], mode, 1.0)
                self.assertEqual(len(view.edges), 1)
                edge = view.edges[0]
                self.assertEqual(
                    (edge.source, edge.target, edge.index),
                    (4, 5, 3),
                )

    def test_short_prefixes_remain_unchanged(self):
        for mode in self.MODES:
            for prefix in ([1], [1, 2]):
                with self.subTest(mode=mode, prefix=prefix):
                    view = self.make_view(prefix, mode, 1.0)
                    self.assertEqual(view.sequence, tuple(prefix))
                    self.assertEqual(len(view.edges), len(prefix) - 1)

    def test_zero_budget_preserves_all_transitions(self):
        for mode in self.MODES:
            with self.subTest(mode=mode):
                view = self.make_view([1, 2, 3, 4], mode, 0.0)
                self.assertEqual(
                    [edge.index for edge in view.edges],
                    [0, 1, 2],
                )

    def test_python_ties_to_even_rounding(self):
        for mode in self.MODES:
            with self.subTest(mode=mode):
                # round(0.5) == 0; round(1.5) == 2.
                first = self.make_view([1, 2, 3], mode, 0.25)
                second = self.make_view([1, 2, 3, 4], mode, 0.50)
                self.assertEqual(len(first.edges), 2)
                self.assertEqual(len(second.edges), 1)

    def test_repeated_transitions_count_as_occurrences(self):
        prefix = [1, 2, 1, 2, 1, 2]

        for mode in self.MODES:
            with self.subTest(mode=mode):
                view = self.make_view(prefix, mode, 0.40)
                self.assertEqual(len(view.edges), 3)
                indices = [edge.index for edge in view.edges]
                self.assertEqual(len(indices), len(set(indices)))
                self.assertIn(4, indices)

    def test_fixed_seed_is_reproducible(self):
        prefix = list(range(1, 22))

        for mode in self.MODES:
            with self.subTest(mode=mode):
                self.assertEqual(
                    self.make_view(prefix, mode, 0.20),
                    self.make_view(prefix, mode, 0.20),
                )

    def test_invalid_budgets_are_rejected(self):
        for mode in self.MODES:
            for budget in (-0.1, 1.1, float("nan"), float("inf")):
                with self.subTest(mode=mode, budget=budget):
                    with self.assertRaises(ValueError):
                        self.make_view([1, 2, 3], mode, budget)

    def test_budgeted_two_views_match_direct_generation(self):
        prefix = list(range(1, 12))

        for mode in self.MODES:
            with self.subTest(mode=mode):
                rng = random.Random(2026)
                expected = (
                    augment_budgeted_edge_view(
                        prefix,
                        mode=mode,
                        budget=0.20,
                        gamma=2.0,
                        rng=rng,
                    ),
                    augment_budgeted_edge_view(
                        prefix,
                        mode=mode,
                        budget=0.20,
                        gamma=2.0,
                        rng=rng,
                    ),
                )
                actual = make_two_views(
                    prefix,
                    mode=mode,
                    budget=0.20,
                    gamma=2.0,
                    seed=2026,
                )
                self.assertEqual(actual, expected)

    def test_probability_two_views_preserve_existing_behavior(self):
        prefix = list(range(1, 12))

        for mode in ("uniform_edge", "recency", "uniform_node"):
            with self.subTest(mode=mode):
                rng = random.Random(2026)
                expected = (
                    augment_session(
                        prefix,
                        mode=mode,
                        probability=0.20,
                        gamma=2.0,
                        rng=rng,
                    ),
                    augment_session(
                        prefix,
                        mode=mode,
                        probability=0.20,
                        gamma=2.0,
                        rng=rng,
                    ),
                )
                actual = make_two_views(
                    prefix,
                    mode=mode,
                    probability=0.20,
                    gamma=2.0,
                    seed=2026,
                )
                self.assertEqual(actual, expected)

    def test_two_view_options_are_validated(self):
        cases = [
            {"mode": "uniform_edge"},
            {
                "mode": "uniform_edge",
                "probability": 0.10,
                "budget": 0.20,
            },
            {"mode": "uniform_node", "budget": 0.20},
            {
                "mode": "uniform_edge",
                "budget": 0.20,
                "seed": 2026,
                "rng": random.Random(2026),
            },
        ]

        for options in cases:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    make_two_views([1, 2, 3, 4], **options)

    def test_budgeted_prefix_validation_uses_shared_rules(self):
        for mode in self.MODES:
            for prefix in (
                [],
                [0, 1],
                [-1, 2],
                [True, 2],
                [1.5, 2],
                ["1", 2],
            ):
                with self.subTest(mode=mode, prefix=prefix):
                    with self.assertRaises(ValueError):
                        self.make_view(prefix, mode, 0.20)

    def test_nearest_count_preserves_existing_rule(self):
        rng = random.Random(2026)
        original_state = rng.getstate()

        for m in range(31):
            for budget in (0.0, 0.10, 0.20, 0.50, 1.0):
                expected = (
                    min(round(budget * m), m - 1)
                    if m > 1 else 0
                )
                self.assertEqual(
                    count_edge_deletions(m, budget, rng=rng),
                    expected,
                )

        self.assertEqual(rng.getstate(), original_state)

    def test_stochastic_count_uses_fractional_probability(self):
        rng = random.Random(2026)

        for budget, draw, expected in (
            (0.10, 0.49, 1),
            (0.10, 0.50, 0),
            (0.30, 0.49, 2),
            (0.30, 0.50, 1),
        ):
            with self.subTest(budget=budget, draw=draw):
                with patch.object(rng, "random", return_value=draw):
                    actual = count_edge_deletions(
                        5,
                        budget,
                        budget_rounding="stochastic",
                        rng=rng,
                    )
                self.assertEqual(actual, expected)

    def test_stochastic_counts_match_across_views_and_policies(self):
        def generate_counts(mode):
            edge_rng = random.Random(17)
            budget_rng = random.Random(2026)
            counts = []

            for length in (1, 2, 3, 6, 8, 11) * 5:
                prefix = list(range(1, length + 1))
                first, second = make_two_views(
                    prefix,
                    mode=mode,
                    budget=0.10,
                    budget_rounding="stochastic",
                    budget_rng=budget_rng,
                    gamma=2.0,
                    rng=edge_rng,
                )
                self.assertEqual(len(first.edges), len(second.edges))
                self.assertEqual(first.sequence, tuple(prefix))
                self.assertEqual(second.sequence, tuple(prefix))

                if length > 1:
                    for view in (first, second):
                        self.assertIn(
                            length - 2,
                            [edge.index for edge in view.edges],
                        )

                counts.append(length - 1 - len(first.edges))

            return counts

        uniform = generate_counts("uniform_edge")
        self.assertEqual(uniform, generate_counts("uniform_edge"))
        self.assertEqual(uniform, generate_counts("recency"))

    def test_stochastic_rounding_keeps_short_prefixes_and_caps_count(self):
        rng = random.Random(2026)

        for m in range(11):
            actual = count_edge_deletions(
                m,
                1.0,
                budget_rounding="stochastic",
                rng=rng,
            )
            self.assertEqual(actual, max(0, m - 1))

    def test_stochastic_requires_separate_budget_rng(self):
        with self.assertRaises(ValueError):
            make_two_views(
                [1, 2, 3, 4, 5, 6],
                mode="uniform_edge",
                budget=0.10,
                budget_rounding="stochastic",
                seed=2026,
            )

        shared_rng = random.Random(2026)
        with self.assertRaises(ValueError):
            make_two_views(
                [1, 2, 3, 4, 5, 6],
                mode="uniform_edge",
                budget=0.10,
                budget_rounding="stochastic",
                rng=shared_rng,
                budget_rng=shared_rng,
            )

class RecencySubstitutionTests(unittest.TestCase):
    def test_exact_changes_training_vocab_and_protected_final_click(self):
        prefix = (1, 2, 3, 4, 5, 6)
        for seed in range(10):
            pair = make_two_views(
                prefix, mode="recency_substitution", budget=0.2,
                gamma=2.0, seed=seed, num_items=12, target=12,
            )
            self.assertEqual(pair, make_two_views(
                prefix, mode="recency_substitution", budget=0.2,
                gamma=2.0, seed=seed, num_items=12, target=12,
            ))
            for view in pair:
                self.assertEqual(sum(a != b for a, b in zip(prefix, view.sequence)), 1)
                self.assertEqual(view.sequence[-1], prefix[-1])
                self.assertTrue(all(1 <= item <= 12 for item in view.sequence))
                self.assertNotIn(12, view.sequence)
                self.assertEqual(
                    [(edge.source, edge.target) for edge in view.edges],
                    list(zip(view.sequence, view.sequence[1:])),
                )

    def test_weights_prefer_older_positions_and_final_position_is_ineligible(self):
        source = random.Random(3)
        original_choices = source.choices
        observed_weights = []

        def choose(population, *, weights, k):
            observed_weights.append(tuple(weights))
            return original_choices(population, weights=weights, k=k)

        with patch.object(source, "choices", side_effect=choose):
            view = augment_recency_substitution(
                [1, 2, 3, 4], num_items=8, count=1, rng=source, gamma=2.0,
            )
        weights = observed_weights[0]
        self.assertEqual(len(weights), 3)
        self.assertGreater(weights[0], weights[1])
        self.assertGreater(weights[1], weights[2])
        self.assertEqual(view.sequence[-1], 4)

    def test_short_prefix_safeguards_match_existing_budget_policy(self):
        for prefix in ([1], [1, 2]):
            first, second = make_two_views(
                prefix, mode="recency_substitution", budget=1.0,
                num_items=5, seed=7,
            )
            self.assertEqual(first.sequence, tuple(prefix))
            self.assertEqual(first, second)

    def test_impossible_replacement_keeps_valid_input(self):
        view = augment_recency_substitution(
            [1, 1, 1], num_items=1, count=1, rng=random.Random(1),
        )
        self.assertEqual(view.sequence, (1, 1, 1))


if __name__ == "__main__":
    unittest.main()
