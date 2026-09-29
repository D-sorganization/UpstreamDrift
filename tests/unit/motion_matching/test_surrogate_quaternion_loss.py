"""Tests for the sign-invariant quaternion supervision loss."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch
from src.shared.python.motion_matching.surrogate._quaternion_loss import (
    quaternion_loss,
)


@pytest.mark.unit
def test_quaternion_loss_zero_for_q_and_neg_q() -> None:
    """L_quat(q, q) == L_quat(q, -q) == 0 — the sign-invariance property."""
    q = torch.tensor([[0.5, 0.5, 0.5, 0.5]])  # already unit-norm
    assert quaternion_loss(q, q).item() == pytest.approx(0.0, abs=1.0e-7)
    assert quaternion_loss(q, -q).item() == pytest.approx(0.0, abs=1.0e-7)


@pytest.mark.unit
def test_quaternion_loss_positive_for_orthogonal() -> None:
    """Two orthogonal quaternions yield the maximum loss of 1.0."""
    q1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    q2 = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    assert quaternion_loss(q1, q2).item() == pytest.approx(1.0, abs=1.0e-6)


@pytest.mark.unit
def test_quaternion_loss_shape_mismatch_raises() -> None:
    """Mismatched shapes are a ValueError, not a silent broadcast."""
    q = torch.zeros((2, 4))
    other = torch.zeros((3, 4))
    with pytest.raises(ValueError):
        quaternion_loss(q, other)


@pytest.mark.unit
def test_quaternion_loss_wrong_last_dim_raises() -> None:
    """Last dim != 4 must raise."""
    q = torch.zeros((2, 3))
    with pytest.raises(ValueError):
        quaternion_loss(q, q)


@pytest.mark.unit
def test_quaternion_loss_differentiable() -> None:
    """Gradient w.r.t. predicted quaternion is finite."""
    q_pred = torch.tensor([[0.5, 0.5, 0.5, 0.5]], requires_grad=True)
    q_true = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    loss = quaternion_loss(q_pred, q_true)
    loss.backward()
    assert q_pred.grad is not None
    assert torch.isfinite(q_pred.grad).all().item()
