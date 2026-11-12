import unittest

import torch

from recsys.data.session_graph import build_session_graph, collate_session_graphs


class SessionGraphTests(unittest.TestCase):
    def test_chain_graph_has_expected_aliases_and_connections(self) -> None:
        graph = build_session_graph([1, 2, 3])

        self.assertEqual(graph.node_ids.tolist(), [1, 2, 3])
        self.assertEqual(graph.alias_indices.tolist(), [0, 1, 2])
        expected = torch.tensor(
            [
                [0, 0, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 0, 0],
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(graph.adjacency, expected)

    def test_repeated_items_share_nodes_and_edges_are_binary(self) -> None:
        graph = build_session_graph([1, 2, 1, 2, 3])

        self.assertEqual(graph.node_ids.tolist(), [1, 2, 3])
        self.assertEqual(graph.alias_indices.tolist(), [0, 1, 0, 1, 2])

        incoming, outgoing = graph.adjacency[:, :3], graph.adjacency[:, 3:]
        # The repeated 1->2 transition remains a single binary connection.
        self.assertEqual(outgoing[0, 1].item(), 1.0)
        self.assertEqual(incoming[1, 0].item(), 1.0)
        self.assertEqual(outgoing[1, 0].item(), 0.5)
        self.assertEqual(outgoing[1, 2].item(), 0.5)

    def test_multiple_predecessors_are_incoming_normalized(self) -> None:
        graph = build_session_graph([1, 3, 2, 3])
        incoming = graph.adjacency[:, :3]

        # Sorted nodes are [1, 2, 3]. Node 3 has predecessors 1 and 2.
        self.assertEqual(incoming[2, 0].item(), 0.5)
        self.assertEqual(incoming[2, 1].item(), 0.5)
        self.assertEqual(incoming[2].sum().item(), 1.0)

    def test_invalid_session_ids_are_rejected(self) -> None:
        for sequence in ([], [0], [-1, 2], [True], [1.5]):
            with self.subTest(sequence=sequence):
                with self.assertRaises(ValueError):
                    build_session_graph(sequence)


class SessionGraphBatchTests(unittest.TestCase):
    def test_collator_pads_nodes_sequences_and_adjacency_channels(self) -> None:
        batch = collate_session_graphs(
            [([5, 7, 5, 9], 2), ([4, 8], 6)]
        )

        self.assertEqual(batch.node_ids.tolist(), [[5, 7, 9], [4, 8, 0]])
        self.assertEqual(batch.alias_indices.tolist(), [[0, 1, 0, 2], [0, 1, 0, 0]])
        self.assertEqual(
            batch.sequence_mask.tolist(),
            [[True, True, True, True], [True, True, False, False]],
        )
        self.assertEqual(batch.targets.tolist(), [2, 6])
        self.assertEqual(tuple(batch.adjacency.shape), (2, 3, 6))

        second_graph = build_session_graph([4, 8])
        torch.testing.assert_close(
            batch.adjacency[1, :2, :2], second_graph.adjacency[:, :2]
        )
        torch.testing.assert_close(
            batch.adjacency[1, :2, 3:5], second_graph.adjacency[:, 2:]
        )
        self.assertEqual(batch.adjacency[1, :, 2].count_nonzero().item(), 0)

    def test_empty_batch_and_invalid_target_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            collate_session_graphs([])
        with self.assertRaises(ValueError):
            collate_session_graphs([([1], 0)])


if __name__ == "__main__":
    unittest.main()
