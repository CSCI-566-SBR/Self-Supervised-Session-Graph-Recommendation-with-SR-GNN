import random
import unittest

from recsys.data.graph_augmentations import augment_session, make_two_views


class GraphAugmentationTests(unittest.TestCase):
    def test_probability_zero_preserves_graph(self) -> None:
        view = augment_session(
            [1, 2, 3], mode="uniform_edge", probability=0.0, rng=random.Random(1)
        )
        self.assertEqual([(edge.source, edge.target) for edge in view.edges], [(1, 2), (2, 3)])

    def test_node_dropout_does_not_invent_shortcut_edge(self) -> None:
        view = augment_session(
            [1, 2, 3, 4], mode="uniform_node", probability=1.0, rng=random.Random(1)
        )
        self.assertEqual(view.sequence, (3, 4))
        self.assertEqual([(edge.source, edge.target) for edge in view.edges], [(3, 4)])
        self.assertNotIn((1, 4), [(edge.source, edge.target) for edge in view.edges])

    def test_recency_protects_final_incoming_edge(self) -> None:
        view = augment_session(
            [1, 2, 3], mode="recency", probability=1.0,
            gamma=1.0, rng=random.Random(1),
        )
        self.assertIn((2, 3), [(edge.source, edge.target) for edge in view.edges])

    def test_short_session_is_unchanged(self) -> None:
        for mode in ("uniform_edge", "uniform_node", "recency"):
            with self.subTest(mode=mode):
                view = augment_session([5, 6], mode=mode, probability=1.0, rng=random.Random(1))
                self.assertEqual(view.sequence, (5, 6))
                self.assertEqual(len(view.edges), 1)

    def test_two_views_are_reproducible(self) -> None:
        first = make_two_views([1, 2, 3, 4, 5, 6], mode="uniform_edge", probability=0.5, seed=2026)
        second = make_two_views([1, 2, 3, 4, 5, 6], mode="uniform_edge", probability=0.5, seed=2026)
        self.assertEqual(first, second)
        self.assertNotEqual(first[0], first[1])


if __name__ == "__main__":
    unittest.main()
