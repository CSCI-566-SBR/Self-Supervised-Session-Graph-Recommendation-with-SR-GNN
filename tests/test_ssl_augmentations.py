import random
import unittest

from recsys.data.graph_augmentations import augment_budgeted_edge_view


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


if __name__ == "__main__":
    unittest.main()