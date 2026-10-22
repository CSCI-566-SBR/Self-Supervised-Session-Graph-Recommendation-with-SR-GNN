import unittest
from math import log2

from recsys.evaluation import (
    format_results,
    get_target_ranks,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


class TestTargetRanks(unittest.TestCase):
    def test_scores_map_to_item_ids_not_column_numbers(self):
        item_ids = [40, 10, 30, 20]
        scores = [
            [0.1, 0.9, 0.3, 0.5],
            [0.8, 0.2, 0.4, 0.6],
            [0.9, 0.8, 0.7, 0.6],
        ]
        self.assertEqual(get_target_ranks(scores, [10, 30, 20], item_ids), [1, 3, 4])

    def test_ties_use_item_ids_independently_of_column_order(self):
        scores = [[0.5, 0.5, 0.5]] * 3
        targets = [30, 10, 20]
        self.assertEqual(get_target_ranks(scores, targets, [30, 10, 20]), [3, 1, 2])
        self.assertEqual(get_target_ranks(scores, targets, [20, 30, 10]), [3, 1, 2])

    def test_negative_scores_and_single_candidate(self):
        self.assertEqual(get_target_ranks([[-1.0, -2.0]], [7], [5, 7]), [2])
        self.assertEqual(get_target_ranks([[0.0]], [7], [7]), [1])

    def test_batch_ranks_can_be_combined_before_averaging(self):
        item_ids = [10, 20, 30]
        scores = [[3.0, 2.0, 1.0]] * 3
        targets = [10, 20, 30]
        ranks = get_target_ranks(scores[:1], targets[:1], item_ids)
        ranks.extend(get_target_ranks(scores[1:], targets[1:], item_ids))
        self.assertEqual(ranks, get_target_ranks(scores, targets, item_ids))
        self.assertAlmostEqual(recall_at_k(ranks, k=1), 1 / 3)

    def test_invalid_inputs(self):
        cases = [
            ([], [], [10], "at least one example"),
            ([[1.0]], [], [10], "same number"),
            ([[]], [10], [], "at least one candidate"),
            ([[1.0]], [10], [10, 20], "one score per item"),
            ([[1.0, 2.0]], [10], [10, 10], "unique"),
            ([[1.0]], [20], [10], "absent"),
            ([[1.0]], [10], [10.0], "integer IDs"),
            ([[1.0]], [10], [True], "integer IDs"),
            ([[1.0]], [10.0], [10], "integer ID"),
            ([[1.0]], [True], [10], "integer ID"),
        ]
        for scores, targets, item_ids, message in cases:
            with self.subTest(scores=scores, targets=targets, item_ids=item_ids):
                with self.assertRaisesRegex(ValueError, message):
                    get_target_ranks(scores, targets, item_ids)

    def test_invalid_scores_in_target_and_other_columns(self):
        for invalid in [float("nan"), float("inf"), -float("inf"), "bad"]:
            for row in [[invalid, 1.0], [1.0, invalid]]:
                with self.subTest(row=row):
                    with self.assertRaisesRegex(ValueError, "finite number"):
                        get_target_ranks([row], [10], [10, 20])


class TestMetrics(unittest.TestCase):
    def test_hand_calculated_average_includes_misses(self):
        ranks = [1, 3, 25]
        self.assertAlmostEqual(recall_at_k(ranks), 2 / 3)
        self.assertAlmostEqual(mrr_at_k(ranks), 4 / 9)
        self.assertAlmostEqual(ndcg_at_k(ranks), 0.5)
        self.assertEqual(ranks, [1, 3, 25])

    def test_cutoff_is_inclusive_and_ranks_are_reusable(self):
        ranks = [10, 11, 20, 21]
        self.assertEqual(recall_at_k(ranks, k=10), 1 / 4)
        self.assertAlmostEqual(mrr_at_k(ranks, k=10), (1 / 10) / 4)
        self.assertAlmostEqual(ndcg_at_k(ranks, k=10), (1 / log2(11)) / 4)
        self.assertEqual(recall_at_k(ranks, k=20), 3 / 4)
        self.assertAlmostEqual(mrr_at_k(ranks, k=20), (1 / 10 + 1 / 11 + 1 / 20) / 4)
        self.assertAlmostEqual(
            ndcg_at_k(ranks, k=20),
            (1 / log2(11) + 1 / log2(12) + 1 / log2(21)) / 4,
        )

    def test_perfect_predictions_and_all_misses(self):
        for metric in [recall_at_k, mrr_at_k, ndcg_at_k]:
            with self.subTest(metric=metric.__name__):
                self.assertEqual(metric([1, 1], k=1), 1.0)
                self.assertEqual(metric([2, 3], k=1), 0.0)

    def test_cutoff_larger_than_candidate_count(self):
        ranks = get_target_ranks([[0.5, 0.9]], [10], [10, 20])
        self.assertEqual(recall_at_k(ranks), 1.0)
        self.assertEqual(mrr_at_k(ranks), 0.5)
        self.assertAlmostEqual(ndcg_at_k(ranks), 1 / log2(3))

    def test_empty_or_invalid_ranks(self):
        for metric in [recall_at_k, mrr_at_k, ndcg_at_k]:
            for ranks in [[], [0], [-1], [1.5], [True], ["1"]]:
                with self.subTest(metric=metric.__name__, ranks=ranks):
                    with self.assertRaisesRegex(ValueError, "ranks"):
                        metric(ranks)

    def test_invalid_cutoffs(self):
        for metric in [recall_at_k, mrr_at_k, ndcg_at_k]:
            for k in [0, -1, 1.5, True, "20"]:
                with self.subTest(metric=metric.__name__, k=k):
                    with self.assertRaisesRegex(ValueError, "k must"):
                        metric([1, 2], k=k)


class TestReporting(unittest.TestCase):
    def test_selected_metrics_in_caller_order(self):
        results = {"MRR@20": 4 / 9, "Recall@20": 2 / 3}
        self.assertEqual(format_results(results), "MRR@20: 44.44%\nRecall@20: 66.67%")

    def test_single_metric_and_empty_results(self):
        self.assertEqual(format_results({"NDCG@10": 0.5}), "NDCG@10: 50.00%")
        self.assertEqual(format_results({}), "")

    def test_invalid_metric_values(self):
        for value in [-0.1, 1.1, float("nan"), float("inf"), "bad"]:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                    format_results({"Recall@20": value})


if __name__ == "__main__":
    unittest.main()
