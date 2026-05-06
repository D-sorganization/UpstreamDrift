"""Private 1D-Transformer encoder block for the inverse CVAE.

A small wrapper around :class:`torch.nn.TransformerEncoder` that handles the
``(batch, seq_len, feature_dim)`` -> ``(batch, seq_len, d_model)`` projection
plus an optional sinusoidal positional encoding. Keeping it private lets the
CVAE module stay focused on the probabilistic plumbing.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def _sinusoidal_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    """Return a ``(seq_len, d_model)`` sinusoidal positional encoding tensor.

    Standard "Attention Is All You Need" formulation. Computed eagerly at
    construction time and registered as a buffer so it follows the module's
    device.
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if d_model <= 0:
        raise ValueError("d_model must be positive")
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    if d_model > 1:
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe


class TransformerSequenceEncoder(nn.Module):
    """1D-Transformer encoder over a fixed-feature kinematic sequence.

    Projects ``(B, T, in_features)`` to ``(B, T, d_model)``, adds a sinusoidal
    positional encoding, runs a stack of ``nn.TransformerEncoderLayer`` blocks,
    and returns the per-timestep hidden states. Pooling (e.g. mean over time)
    is left to the caller so that the same encoder can be used for both context
    summaries and posterior conditioning.

    Parameters
    ----------
    in_features
        Number of input channels per timestep.
    d_model
        Hidden width of the transformer.
    n_heads
        Number of attention heads. Must divide ``d_model``.
    n_layers
        Number of stacked encoder layers.
    dropout
        Dropout on attention and feed-forward sub-layers.
    max_seq_len
        Pre-allocated positional-encoding length. The forward pass tolerates
        any ``T <= max_seq_len`` and re-allocates lazily for longer sequences.
    """

    def __init__(
        self,
        *,
        in_features: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        if n_layers <= 0:
            raise ValueError("n_layers must be positive")
        self.in_features = in_features
        self.d_model = d_model
        self.input_proj = nn.Linear(in_features, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        pe = _sinusoidal_positional_encoding(max_seq_len, d_model)
        self._pe: torch.Tensor
        self.register_buffer("_pe", pe, persistent=False)

    def _positional_encoding(self, seq_len: int) -> torch.Tensor:
        """Return a positional encoding sized to ``seq_len``.

        Re-allocates lazily (and on the correct device) if the cached buffer
        is too short for the request.
        """
        cached: torch.Tensor = self._pe
        if seq_len <= cached.shape[0]:
            return cached[:seq_len]
        pe = _sinusoidal_positional_encoding(seq_len, self.d_model).to(cached.device)
        self._pe = pe
        return pe

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the transformer encoder over a kinematic sequence.

        Parameters
        ----------
        x
            ``(B, T, in_features)`` float tensor.
        attention_mask
            Optional ``(B, T)`` bool mask where ``True`` marks padding tokens
            to ignore. Forwarded to ``TransformerEncoder`` as
            ``src_key_padding_mask``.

        Returns
        -------
        torch.Tensor
            ``(B, T, d_model)`` hidden states.
        """
        if x.dim() != 3:
            raise ValueError(f"x must be 3D (B, T, F); got {tuple(x.shape)}")
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"x last-dim must be {self.in_features}; got {x.shape[-1]}"
            )
        seq_len = x.shape[1]
        h = self.input_proj(x) + self._positional_encoding(seq_len)
        return self.encoder(h, src_key_padding_mask=attention_mask)
