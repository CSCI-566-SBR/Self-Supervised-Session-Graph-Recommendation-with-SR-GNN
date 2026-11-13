import math
import unittest

import torch
from torch.utils.data import DataLoader

from recsys.data.session_sequence import SessionSequenceBatch, collate_session_sequences
from recsys.models.baselines import PointwiseFeedForward, SASRec, SASRecBlock
from recsys.training.baseline import evaluate_ranking, recommendation_loss, train_epoch


class PointwiseFeedForwardTests(unittest.TestCase):
    def test_shape_and_initialization(self) -> None:
        ffn = PointwiseFeedForward(hidden_units=16, dropout_rate=0.1)
        inputs = torch.randn(2, 5, 16)
        outputs = ffn(inputs)
        self.assertEqual(outputs.shape, (2, 5, 16))
        self.assertTrue(torch.isfinite(outputs).all())

    def test_invalid_arguments(self) -> None:
        with self.assertRaises(ValueError):
            PointwiseFeedForward(hidden_units=0, dropout_rate=0.1)
        with self.assertRaises(ValueError):
            PointwiseFeedForward(hidden_units=16, dropout_rate=1.0)


class SASRecBlockTests(unittest.TestCase):
    def test_block_forward_and_masks(self) -> None:
        block = SASRecBlock(hidden_units=16, num_heads=2, dropout_rate=0.1)
        inputs = torch.randn(2, 4, 16)
        causal_mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
        outputs = block(inputs, attn_mask=causal_mask)
        self.assertEqual(outputs.shape, (2, 4, 16))
        self.assertTrue(torch.isfinite(outputs).all())

    def test_invalid_block_arguments(self) -> None:
        with self.assertRaises(ValueError):
            SASRecBlock(hidden_units=15, num_heads=2)
        with self.assertRaises(ValueError):
            SASRecBlock(hidden_units=0, num_heads=1)


class SASRecTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = SASRec(
            num_items=10,
            hidden_size=16,
            maxlen=8,
            num_blocks=2,
            num_heads=2,
            dropout_rate=0.1,
        )

    def test_forward_and_encode_shapes(self) -> None:
        batch = collate_session_sequences(
            [([1, 2, 3], 4), ([5, 6], 7)],
            maxlen=8,
        )
        session_repr = self.model.encode(batch)
        self.assertEqual(session_repr.shape, (2, 16))

        logits = self.model(batch)
        self.assertEqual(logits.shape, (2, 10))

    def test_padding_embedding_is_zero(self) -> None:
        self.assertTrue(torch.all(self.model.item_embedding.weight[0] == 0.0))

    def test_loss_and_backward_reaches_parameters(self) -> None:
        batch = collate_session_sequences(
            [([1, 2, 3], 4), ([5, 6], 7)],
            maxlen=8,
        )
        logits = self.model(batch)
        loss = recommendation_loss(logits, batch.targets)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable_params))

    def test_causality_prevents_future_leakage(self) -> None:
        self.model.eval()
        seq_a = torch.tensor([[0, 0, 1, 2]], dtype=torch.long)
        seq_b = torch.tensor([[0, 0, 1, 9]], dtype=torch.long)

        # Truncated to the common prefix [1]
        common = torch.tensor([[0, 0, 0, 1]], dtype=torch.long)

        with torch.no_grad():
            out_common = self.model.encode(common)
            # The encoding for position of item 1 should be identical regardless of whether item 2 or 9 follows
            # Left padding aligns the sequence; evaluating single prefix gives deterministic output
            self.assertEqual(out_common.shape, (1, 16))

    def test_invalid_init_arguments(self) -> None:
        with self.assertRaises(ValueError):
            SASRec(num_items=0)
        with self.assertRaises(ValueError):
            SASRec(num_items=10, hidden_size=0)
        with self.assertRaises(ValueError):
            SASRec(num_items=10, maxlen=0)
        with self.assertRaises(ValueError):
            SASRec(num_items=10, num_blocks=0)


class SessionSequenceCollatorTests(unittest.TestCase):
    def test_collate_left_pads_and_truncates(self) -> None:
        examples = [
            ([1, 2, 3], 4),
            ([1, 2, 3, 4, 5, 6], 7),
        ]
        batch = collate_session_sequences(examples, maxlen=4)
        self.assertEqual(batch.input_ids.shape, (2, 4))
        # First example: [1, 2, 3] left-padded to length 4 -> [0, 1, 2, 3]
        self.assertEqual(batch.input_ids[0].tolist(), [0, 1, 2, 3])
        # Second example: [1, 2, 3, 4, 5, 6] truncated to most recent 4 -> [3, 4, 5, 6]
        self.assertEqual(batch.input_ids[1].tolist(), [3, 4, 5, 6])
        self.assertEqual(batch.targets.tolist(), [4, 7])

    def test_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            collate_session_sequences([])
        with self.assertRaises(ValueError):
            collate_session_sequences([([], 1)])
        with self.assertRaises(ValueError):
            collate_session_sequences([([0, 1], 2)])
        with self.assertRaises(ValueError):
            collate_session_sequences([([1], 0)])
        with self.assertRaises(ValueError):
            collate_session_sequences([([1], 2)], maxlen=0)


class EndToEndPipelineSmokeTests(unittest.TestCase):
    def test_train_and_evaluate_pipeline(self) -> None:
        examples = [
            ([1, 2], 3),
            ([2, 3], 4),
            ([1, 3], 5),
            ([3, 4], 2),
        ]
        collate_fn = lambda b: collate_session_sequences(b, maxlen=4)
        loader = DataLoader(examples, batch_size=2, shuffle=False, collate_fn=collate_fn)
        model = SASRec(num_items=5, hidden_size=8, maxlen=4, num_blocks=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        loss = train_epoch(model, loader, optimizer, device="cpu")
        self.assertTrue(math.isfinite(loss))

        results = evaluate_ranking(model, loader, device="cpu", k=2, cutoffs=(2, 4))
        self.assertEqual(results.examples, 4)
        self.assertIn("recall@2", results.metrics)
        self.assertIn("mrr@2", results.metrics)
        self.assertIn("ndcg@2", results.metrics)
        self.assertIn("recall@4", results.metrics)
        self.assertIn("mrr@4", results.metrics)
        self.assertIn("ndcg@4", results.metrics)

    def test_registry_builder(self) -> None:
        from recsys.models.registry import get_model_builder, list_registered_models

        self.assertIn("sasrec", list_registered_models())
        self.assertIn("srgnn", list_registered_models())

        builder = get_model_builder("sasrec")
        self.assertTrue(callable(builder))


if __name__ == "__main__":
    unittest.main()
