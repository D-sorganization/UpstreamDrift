"""Unit tests for the private 1D-Transformer encoder helper."""

from __future__ import annotations

import pytest
import torch
from src.shared.python.motion_matching.inverse._transformer import (
    TransformerSequenceEncoder,
)


def _make_encoder(**overrides: object) -> TransformerSequenceEncoder:
    """Build a small TransformerSequenceEncoder for tests."""
    base: dict[str, object] = {
        "in_features": 12,
        "d_model": 16,
        "n_heads": 2,
        "n_layers": 2,
        "dropout": 0.0,
        "max_seq_len": 64,
    }
    base.update(overrides)
    return TransformerSequenceEncoder(**base)  # type: ignore[arg-type]


@pytest.mark.unit
def test_transformer_attention_mask_optional() -> None:
    enc = _make_encoder().eval()
    x = torch.randn(3, 20, 12)

    out_no_mask = enc(x)
    mask = torch.zeros(3, 20, dtype=torch.bool)
    mask[:, -5:] = True  # mark trailing 5 timesteps as padding
    out_masked = enc(x, attention_mask=mask)

    assert out_no_mask.shape == (3, 20, 16)
    assert out_masked.shape == (3, 20, 16)
    # Different masks give different outputs (otherwise mask wiring is broken).
    assert not torch.allclose(out_no_mask, out_masked)


@pytest.mark.unit
def test_transformer_supports_arbitrary_seq_len() -> None:
    enc = _make_encoder(max_seq_len=32).eval()

    for seq_len in (1, 7, 32, 64):  # 64 > max_seq_len -> triggers re-alloc
        x = torch.randn(2, seq_len, 12)
        out = enc(x)
        assert out.shape == (2, seq_len, 16)
        assert torch.isfinite(out).all()


@pytest.mark.unit
def test_transformer_output_finite_for_zero_input() -> None:
    enc = _make_encoder().eval()
    x = torch.zeros(2, 10, 12)

    out = enc(x)

    assert out.shape == (2, 10, 16)
    assert torch.isfinite(out).all()


@pytest.mark.unit
def test_transformer_rejects_invalid_shapes_and_config() -> None:
    enc = _make_encoder().eval()

    # Wrong rank.
    with pytest.raises(ValueError):
        enc(torch.randn(4, 12))
    # Wrong feature dim.
    with pytest.raises(ValueError):
        enc(torch.randn(2, 5, 7))
    # n_heads must divide d_model.
    with pytest.raises(ValueError):
        TransformerSequenceEncoder(in_features=12, d_model=15, n_heads=4, n_layers=1)
    # n_layers must be positive.
    with pytest.raises(ValueError):
        TransformerSequenceEncoder(in_features=12, d_model=16, n_heads=2, n_layers=0)
