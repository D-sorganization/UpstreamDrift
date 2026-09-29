"""Tests for the SwingSurrogate FiLM-MLP module."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch
from src.shared.python.motion_matching.surrogate.model import (
    ClubTrajectory,
    SurrogateConfig,
    SwingSurrogate,
)


@pytest.fixture
def small_cfg() -> SurrogateConfig:
    """A small SurrogateConfig that runs in milliseconds on CPU."""
    return SurrogateConfig(
        n_joints=4,
        coeffs_per_joint=7,
        seq_len=20,
        hidden_dim=32,
        n_layers=2,
        time_embed_dim=16,
        encoder_layers=2,
    )


@pytest.fixture
def small_model(small_cfg: SurrogateConfig) -> SwingSurrogate:
    """Build the surrogate from ``small_cfg`` with a deterministic seed."""
    torch.manual_seed(0)
    return SwingSurrogate(small_cfg)


@pytest.mark.unit
def test_forward_returns_correct_output_shapes(
    small_model: SwingSurrogate, small_cfg: SurrogateConfig
) -> None:
    """forward() must produce a ClubTrajectory with the documented shapes."""
    batch = 3
    coeffs = torch.randn(batch, small_cfg.coeff_dim)
    out = small_model(coeffs)
    assert isinstance(out, ClubTrajectory)
    assert out.butt.shape == (batch, small_cfg.seq_len, 3)
    assert out.clubhead.shape == (batch, small_cfg.seq_len, 3)
    assert out.club_quat.shape == (batch, small_cfg.seq_len, 4)
    assert out.joint_q.shape == (batch, small_cfg.seq_len, small_cfg.n_joints)


@pytest.mark.unit
def test_surrogate_output_quaternions_unit_norm(
    small_model: SwingSurrogate, small_cfg: SurrogateConfig
) -> None:
    """Every emitted quaternion is unit-norm and has w >= 0 (canonicalized)."""
    coeffs = torch.randn(2, small_cfg.coeff_dim)
    out = small_model(coeffs)
    norms = out.club_quat.norm(dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1.0e-5, rtol=1.0e-5)
    assert (out.club_quat[..., 0] >= 0).all().item()


@pytest.mark.unit
def test_surrogate_gradient_finite(
    small_model: SwingSurrogate, small_cfg: SurrogateConfig
) -> None:
    """∂(loss)/∂coeffs must be finite and non-vanishing — Option 2's lifeline."""
    coeffs = torch.randn(1, small_cfg.coeff_dim, requires_grad=True)
    pred = small_model(coeffs)
    loss = pred.butt.pow(2).mean() + pred.clubhead.pow(2).mean()
    loss.backward()
    grad = coeffs.grad
    assert grad is not None
    assert torch.isfinite(grad).all().item(), "gradients have NaN or Inf"
    assert grad.abs().max().item() > 1.0e-10, "gradient is degenerate"


@pytest.mark.unit
def test_surrogate_quaternion_path_differentiable(
    small_model: SwingSurrogate, small_cfg: SurrogateConfig
) -> None:
    """Gradient through the quaternion head must also be finite & non-zero."""
    coeffs = torch.randn(1, small_cfg.coeff_dim, requires_grad=True)
    pred = small_model(coeffs)
    q_loss = (1.0 - pred.club_quat[..., 0]).mean()
    q_loss.backward()
    assert coeffs.grad is not None
    assert torch.isfinite(coeffs.grad).all().item()
    assert coeffs.grad.abs().max().item() > 1.0e-10


@pytest.mark.unit
def test_surrogate_rejects_wrong_input_dim(
    small_model: SwingSurrogate, small_cfg: SurrogateConfig
) -> None:
    """A coeffs tensor with the wrong second-dim should trip the precondition."""
    bad = torch.randn(1, small_cfg.coeff_dim + 1)
    from src.shared.python.core.contracts import PreconditionError

    with pytest.raises(PreconditionError):
        small_model(bad)


@pytest.mark.unit
def test_surrogate_rejects_zero_n_joints() -> None:
    """SurrogateConfig with n_joints=0 must fail at __init__."""
    cfg = SurrogateConfig(n_joints=0)
    with pytest.raises(ValueError):
        SwingSurrogate(cfg)


@pytest.mark.unit
def test_surrogate_rejects_odd_time_embed_dim() -> None:
    """An odd time_embed_dim breaks the sin/cos split — must raise."""
    cfg = SurrogateConfig(n_joints=2, time_embed_dim=15)
    with pytest.raises(ValueError):
        SwingSurrogate(cfg)
