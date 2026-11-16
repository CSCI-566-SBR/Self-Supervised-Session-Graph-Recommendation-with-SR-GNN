"""Self-Attentive Sequential Recommendation (SASRec) baseline.

References:
- Kang & McAuley (ICDM 2018): Self-Attentive Sequential Recommendation
- Shared project evaluation protocol: full-vocabulary candidate ranking.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Callable

import torch
from torch import Tensor, nn

from recsys.data.session_sequence import SessionSequenceBatch, collate_session_sequences
from recsys.models.registry import register_model

if TYPE_CHECKING:
    from recsys.data.processed import ProcessedDataset


class PointwiseFeedForward(nn.Module):
    """Two-layer 1D convolutional point-wise feed-forward network with dropout and ReLU."""

    def __init__(self, hidden_units: int, dropout_rate: float) -> None:
        super().__init__()
        if hidden_units <= 0:
            raise ValueError("hidden_units must be positive")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout_rate must lie in [0, 1)")

        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        # inputs: [B, seq_len, hidden_units] -> transpose to [B, hidden_units, seq_len]
        x = inputs.transpose(1, 2)
        x = self.dropout1(self.relu(self.conv1(x)))
        x = self.dropout2(self.conv2(x))
        return x.transpose(1, 2)


class SASRecBlock(nn.Module):
    """Pre-LN Transformer decoder block with causal self-attention and point-wise FFN."""

    def __init__(
        self,
        hidden_units: int,
        num_heads: int = 1,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        if hidden_units <= 0:
            raise ValueError("hidden_units must be positive")
        if num_heads <= 0 or hidden_units % num_heads != 0:
            raise ValueError("hidden_units must be divisible by positive num_heads")

        self.ln1 = nn.LayerNorm(hidden_units, eps=1e-8)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_units,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.ln2 = nn.LayerNorm(hidden_units, eps=1e-8)
        self.ffn = PointwiseFeedForward(hidden_units, dropout_rate)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.ln1.reset_parameters()
        self.attention._reset_parameters()
        self.ln2.reset_parameters()
        self.ffn.reset_parameters()

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        # Pre-LN Self-Attention
        norm_x = self.ln1(x)
        attn_out, _ = self.attention(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            attn_mask=attn_mask,
            need_weights=False,
        )
        x = x + self.dropout1(attn_out)

        # Pre-LN Feed-Forward
        norm_x = self.ln2(x)
        x = x + self.ffn(norm_x)
        return x


class SASRec(nn.Module):
    """Self-Attentive Sequential Recommendation model adapted for session recommendation.

    Matches the official SASRec architecture with tied item embeddings, learnable
    positional embeddings, and full-vocabulary candidate dot-product scoring.
    """

    def __init__(
        self,
        num_items: int,
        hidden_size: int = 100,
        maxlen: int = 50,
        num_blocks: int = 2,
        num_heads: int = 1,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        if num_items <= 0:
            raise ValueError("num_items must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if maxlen <= 0:
            raise ValueError("maxlen must be positive")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")

        self.num_items = num_items
        self.hidden_size = hidden_size
        self.maxlen = maxlen
        self.num_blocks = num_blocks
        self.num_heads = num_heads

        # Row 0 reserved for padding token. Items are indexed 1..num_items.
        self.item_embedding = nn.Embedding(
            num_items + 1, hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(maxlen, hidden_size)
        self.embedding_dropout = nn.Dropout(p=dropout_rate)

        self.blocks = nn.ModuleList(
            [
                SASRecBlock(
                    hidden_units=hidden_size,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_layernorm = nn.LayerNorm(hidden_size, eps=1e-8)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(maxlen, maxlen, dtype=torch.bool), diagonal=1),
            persistent=False,
        )
        self.register_buffer(
            "positions",
            torch.arange(maxlen, dtype=torch.long),
            persistent=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize all model parameters and zero out padding embedding row."""

        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.position_embedding.weight)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

        for block in self.blocks:
            block.reset_parameters()
        self.final_layernorm.reset_parameters()

    def _extract_input_ids(self, batch: SessionSequenceBatch | Tensor) -> Tensor:
        if isinstance(batch, SessionSequenceBatch):
            return batch.input_ids
        if isinstance(batch, Tensor):
            return batch
        raise TypeError(f"expected SessionSequenceBatch or Tensor, got {type(batch).__name__}")

    def encode(self, batch: SessionSequenceBatch | Tensor) -> Tensor:
        """Encode left-padded sequence into the final contextual session representation."""

        input_ids = self._extract_input_ids(batch)
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch_size, seq_len]")
        batch_size, seq_len = input_ids.shape
        if seq_len > self.maxlen:
            input_ids = input_ids[:, -self.maxlen:]
            seq_len = self.maxlen

        positions = self.positions[:seq_len].unsqueeze(0).expand(batch_size, -1)
        item_vectors = self.item_embedding(input_ids)
        position_vectors = self.position_embedding(positions)
        sequence_hidden = self.embedding_dropout(item_vectors + position_vectors)

        # Causal attention mask slice from pre-allocated buffer
        causal_mask = self.causal_mask[:seq_len, :seq_len]

        for block in self.blocks:
            sequence_hidden = block(
                sequence_hidden,
                attn_mask=causal_mask,
            )

        sequence_hidden = self.final_layernorm(sequence_hidden)
        # Left-padded sequences always terminate at the final sequence position (-1)
        return sequence_hidden[:, -1, :]

    def forward(self, batch: SessionSequenceBatch | Tensor) -> Tensor:
        """Compute full-vocabulary recommendation logits via shared item embeddings."""

        session_embeddings = self.encode(batch)
        candidate_embeddings = self.item_embedding.weight[1:]
        return torch.matmul(session_embeddings, candidate_embeddings.transpose(0, 1))


@register_model("sasrec")
def build_sasrec(
    params: dict[str, Any], dataset: ProcessedDataset
) -> tuple[nn.Module, Callable[[list[Any]], Any]]:
    """Build and validate a SASRec baseline model and its session sequence collator."""

    hidden_size = params.get("hidden_size", 100)
    maxlen = params.get("maxlen", 50)
    num_blocks = params.get("num_blocks", 2)
    num_heads = params.get("num_heads", 1)
    dropout_rate = params.get("dropout_rate", 0.2)

    positive_integers = {
        "hidden_size": hidden_size,
        "maxlen": maxlen,
        "num_blocks": num_blocks,
        "num_heads": num_heads,
    }
    for name, value in positive_integers.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError("dropout_rate must lie in [0, 1)")

    model = SASRec(
        num_items=dataset.num_items,
        hidden_size=hidden_size,
        maxlen=maxlen,
        num_blocks=num_blocks,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
    )
    collate_fn = lambda b: collate_session_sequences(b, maxlen=maxlen)
    return model, collate_fn
