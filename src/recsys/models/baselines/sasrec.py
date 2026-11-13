"""Self-Attentive Sequential Recommendation (SASRec) baseline.

References:
- Kang & McAuley (ICDM 2018): Self-Attentive Sequential Recommendation
  https://arxiv.org/abs/1808.09781

Formulation and Adaptation Notes:
1. Training Loss & Sequence Format:
   - Original SASRec (Kang & McAuley, 2018) is formulated for long user interaction sequences
     trained with sequence-to-sequence next-item binary cross-entropy (BCE) loss using 1 sampled
     negative item per time-step:
       L = -sum_{t} [log(sigma(r_{i_t, t})) + log(1 - sigma(r_{j_t, t}))]
   - Session Adaptation: To ensure rigorous, direct comparability with session-based baselines
     (SR-GNN / Wu et al., 2019) on identical benchmark splits, each session is expanded into prefix
     subsequences [v_1, ..., v_t] predicting the single target item v_{t+1}. Training optimizes
     multiclass cross-entropy across all candidate catalog items:
       L = -log(exp(s_{v_{t+1}}) / sum_{j in V} exp(s_j))
2. Item Embeddings & Scaling:
   - Matches official SASRec by scaling input item embeddings by sqrt(hidden_size) before adding
     learnable positional embeddings.
   - Padded positions (index 0) are explicitly masked during multi-head attention and zeroed out
     after each block and the final LayerNorm.
3. Item Catalog & Indexing:
   - The embedding table contains num_items + 1 rows, where index 0 is reserved for padding.
   - For Diginetica, |V| + 1 = 42,069 embedding rows (index 0 reserved for padding; 42,068 candidate target items).
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

        self.num_heads = num_heads
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
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        # Pre-LN Self-Attention with causal and key-padding masks
        norm_x = self.ln1(x)

        effective_attn_mask = attn_mask
        mha_key_padding_mask = key_padding_mask

        if key_padding_mask is not None and attn_mask is not None:
            # Combine causal mask [S, S] and key_padding_mask [B, S] into [B, S, S]
            if attn_mask.dim() == 2:
                combined_mask = attn_mask.unsqueeze(0) | key_padding_mask.unsqueeze(1)
            else:
                combined_mask = attn_mask.clone() | key_padding_mask.unsqueeze(1)

            # Unmask diagonal for padding queries so no query row is completely masked (prevents NaN softmax)
            seq_len = x.size(1)
            diag_idx = torch.arange(seq_len, device=x.device)
            combined_mask = combined_mask.clone()
            combined_mask[:, diag_idx, diag_idx] = False

            # PyTorch MultiheadAttention expects 3D attn_mask shaped (B * num_heads, S, S)
            effective_attn_mask = combined_mask.repeat_interleave(self.num_heads, dim=0)
            mha_key_padding_mask = None

        attn_out, _ = self.attention(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            attn_mask=effective_attn_mask,
            key_padding_mask=mha_key_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout1(attn_out)

        # Pre-LN Feed-Forward
        norm_x = self.ln2(x)
        x = x + self.ffn(norm_x)

        # Zero out padding representations after each block
        if key_padding_mask is not None:
            x = x * (~key_padding_mask).unsqueeze(-1)

        return x


class SASRec(nn.Module):
    """Self-Attentive Sequential Recommendation model adapted for session recommendation.

    Architecture Details:
    - Pre-LN Transformer blocks with causal self-attention and pointwise feed-forward networks.
    - Tied candidate item embeddings scaled by sqrt(d) and evaluated across the entire catalog.
    - Learnable positional embeddings up to maximum sequence length `maxlen`.
    - Key-padding masking during attention and explicit post-block representation zeroing.
    - Sliced final time-step representation for dot-product scoring against candidate items.

    Note on Training Formulation:
    Unlike Kang & McAuley (2018)'s BCE loss on sampled negatives, this implementation uses
    session prefix-expansion and full-vocabulary cross-entropy loss to match the SR-GNN
    experimental protocol directly.
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

    def encode(
        self,
        batch: SessionSequenceBatch | Tensor,
        return_all_steps: bool = False,
    ) -> Tensor:
        """Encode left-padded sequence into contextual session representations."""

        input_ids = self._extract_input_ids(batch)
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch_size, seq_len]")
        batch_size, seq_len = input_ids.shape
        if seq_len > self.maxlen:
            input_ids = input_ids[:, -self.maxlen:]
            seq_len = self.maxlen

        padding_mask = (input_ids == 0)
        positions = self.positions[:seq_len].unsqueeze(0).expand(batch_size, -1)

        # Scale item embeddings by sqrt(hidden_size) matching official SASRec
        item_vectors = self.item_embedding(input_ids) * math.sqrt(self.hidden_size)
        position_vectors = self.position_embedding(positions)
        sequence_hidden = self.embedding_dropout(item_vectors + position_vectors)
        sequence_hidden = sequence_hidden * (~padding_mask).unsqueeze(-1)

        # Causal attention mask slice from pre-allocated buffer
        causal_mask = self.causal_mask[:seq_len, :seq_len]

        for block in self.blocks:
            sequence_hidden = block(
                sequence_hidden,
                attn_mask=causal_mask,
                key_padding_mask=padding_mask,
            )

        sequence_hidden = self.final_layernorm(sequence_hidden)
        sequence_hidden = sequence_hidden * (~padding_mask).unsqueeze(-1)

        if return_all_steps:
            return sequence_hidden

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
