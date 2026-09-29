"""Unit tests for :class:`TimestepInverseDynamics`."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.shared.python.motion_matching.inverse_timestep.model import (  # noqa: E402
    DEFAULT_INPUT_DIM,
    DEFAULT_OUTPUT_DIM,
    TimestepInverseConfig,
    TimestepInverseDynamics,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_torch]


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_timestep_inverse_model_config_defaults() -> None:
    cfg = TimestepInverseConfig()
    cfg.validate()
    assert cfg.input_dim == DEFAULT_INPUT_DIM == 81
    assert cfg.output_dim == DEFAULT_OUTPUT_DIM == 27
    assert cfg.use_missing_indicator is True
    assert cfg.effective_input_dim == 162


def test_config_rejects_zero_input_dim() -> None:
    with pytest.raises(ValueError, match="input_dim"):
        TimestepInverseConfig(input_dim=0).validate()


def test_config_rejects_negative_n_blocks() -> None:
    with pytest.raises(ValueError, match="n_blocks"):
        TimestepInverseConfig(n_blocks=0).validate()


def test_config_rejects_dropout_one() -> None:
    with pytest.raises(ValueError, match="dropout"):
        TimestepInverseConfig(dropout=1.0).validate()


def test_config_no_indicator_input_doubled_off() -> None:
    cfg = TimestepInverseConfig(use_missing_indicator=False)
    cfg.validate()
    assert cfg.effective_input_dim == cfg.input_dim


# ---------------------------------------------------------------------------
# Forward shape / dtype
# ---------------------------------------------------------------------------


def _make_state(n: int = 4, dim: int = DEFAULT_INPUT_DIM) -> torch.Tensor:
    return torch.zeros(n, dim, dtype=torch.float32)


def test_forward_returns_correct_shape_and_dtype() -> None:
    model = TimestepInverseDynamics()
    state = _make_state(8)
    out = model(state)
    assert out.shape == (8, DEFAULT_OUTPUT_DIM)
    assert out.dtype == torch.float32


def test_forward_with_small_config() -> None:
    cfg = TimestepInverseConfig(hidden=32, n_blocks=2, dropout=0.0)
    model = TimestepInverseDynamics(cfg)
    state = torch.randn(3, cfg.input_dim, dtype=torch.float32)
    out = model(state)
    assert out.shape == (3, cfg.output_dim)


def test_forward_deterministic_in_eval() -> None:
    cfg = TimestepInverseConfig(hidden=32, n_blocks=2, dropout=0.0)
    model = TimestepInverseDynamics(cfg)
    model.eval()
    state = torch.randn(2, cfg.input_dim, dtype=torch.float32)
    with torch.no_grad():
        a = model(state)
        b = model(state)
    torch.testing.assert_close(a, b)


def test_forward_gradient_flow() -> None:
    cfg = TimestepInverseConfig(hidden=32, n_blocks=2, dropout=0.0)
    model = TimestepInverseDynamics(cfg)
    state = torch.randn(4, cfg.input_dim, dtype=torch.float32)
    target = torch.randn(4, cfg.output_dim, dtype=torch.float32)
    pred = model(state)
    loss = ((pred - target) ** 2).mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert any((g.abs().sum() > 0).item() for g in grads if g is not None)


# ---------------------------------------------------------------------------
# Input validation (DbC)
# ---------------------------------------------------------------------------


def test_forward_rejects_non_tensor() -> None:
    model = TimestepInverseDynamics()
    with pytest.raises(TypeError, match="torch.Tensor"):
        model([0.0] * DEFAULT_INPUT_DIM)  # type: ignore[arg-type]


def test_timestep_inverse_model_forward_rejects_wrong_rank() -> None:
    model = TimestepInverseDynamics()
    with pytest.raises(ValueError, match="2-D"):
        model(torch.zeros(DEFAULT_INPUT_DIM, dtype=torch.float32))


def test_forward_rejects_wrong_last_dim() -> None:
    model = TimestepInverseDynamics()
    with pytest.raises(ValueError, match="last-dim"):
        model(torch.zeros(2, DEFAULT_INPUT_DIM - 1, dtype=torch.float32))


def test_forward_rejects_non_float32() -> None:
    model = TimestepInverseDynamics()
    with pytest.raises(TypeError, match="float32"):
        model(torch.zeros(2, DEFAULT_INPUT_DIM, dtype=torch.float64))


def test_forward_rejects_inf_input() -> None:
    model = TimestepInverseDynamics()
    bad = torch.zeros(2, DEFAULT_INPUT_DIM, dtype=torch.float32)
    bad[0, 0] = float("inf")
    with pytest.raises(ValueError, match="Inf"):
        model(bad)


def test_forward_tolerates_nan_input_when_indicator_on() -> None:
    cfg = TimestepInverseConfig(hidden=32, n_blocks=2, dropout=0.0)
    model = TimestepInverseDynamics(cfg)
    state = torch.randn(2, cfg.input_dim, dtype=torch.float32)
    state[0, 0] = float("nan")
    state[1, 5] = float("nan")
    out = model(state)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_forward_tolerates_nan_when_indicator_off() -> None:
    cfg = TimestepInverseConfig(
        hidden=32, n_blocks=2, dropout=0.0, use_missing_indicator=False
    )
    model = TimestepInverseDynamics(cfg)
    state = torch.randn(2, cfg.input_dim, dtype=torch.float32)
    state[0, 0] = float("nan")
    out = model(state)
    assert not torch.isnan(out).any()
