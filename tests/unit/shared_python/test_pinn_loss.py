"""Tests for the PINN loss function module (Phase 2 of the PINNs epic #5419).

Tests are skipped gracefully when JAX is not installed.
"""

from __future__ import annotations

import pytest

try:
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

pytestmark = pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")

from src.shared.python.physics_informed.loss import (  # noqa: E402
    LossWeights,
    contact_loss,
    data_loss,
    physics_loss,
    total_loss,
)

# ---------------------------------------------------------------------------
# LossWeights
# ---------------------------------------------------------------------------


def test_loss_weights_valid() -> None:
    weights = LossWeights(data_weight=1.0, physics_weight=0.5, contact_weight=0.1)
    assert weights.data_weight == 1.0
    assert weights.physics_weight == 0.5
    assert weights.contact_weight == 0.1


def test_loss_weights_rejects_negative() -> None:
    with pytest.raises(ValueError, match="data_weight"):
        LossWeights(data_weight=-1.0, physics_weight=1.0, contact_weight=1.0)


def test_loss_weights_rejects_zero() -> None:
    with pytest.raises(ValueError, match="data_weight"):
        LossWeights(data_weight=0.0, physics_weight=1.0, contact_weight=1.0)


def test_loss_weights_rejects_negative_physics() -> None:
    with pytest.raises(ValueError, match="physics_weight"):
        LossWeights(data_weight=1.0, physics_weight=-0.5, contact_weight=1.0)


def test_loss_weights_rejects_zero_contact() -> None:
    with pytest.raises(ValueError, match="contact_weight"):
        LossWeights(data_weight=1.0, physics_weight=1.0, contact_weight=0.0)


# ---------------------------------------------------------------------------
# data_loss
# ---------------------------------------------------------------------------


def test_data_loss_zero_on_perfect_prediction() -> None:
    x = jnp.array([1.0, 2.0, 3.0])
    assert float(data_loss(x, x)) == pytest.approx(0.0)


def test_data_loss_positive_on_error() -> None:
    pred = jnp.array([1.0, 0.0])
    actual = jnp.array([0.0, 1.0])
    assert float(data_loss(pred, actual)) > 0


def test_data_loss_mse_correctness() -> None:
    pred = jnp.array([2.0])
    actual = jnp.array([0.0])
    # MSE = mean((2 - 0)^2) = 4.0
    assert float(data_loss(pred, actual)) == pytest.approx(4.0)


def test_data_loss_scalar() -> None:
    """Return value must be scalar (0-dimensional)."""
    result = data_loss(jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))
    assert result.ndim == 0


# ---------------------------------------------------------------------------
# physics_loss
# ---------------------------------------------------------------------------


def test_physics_loss_zero_within_limits() -> None:
    torques = jnp.array([0.0, 0.0])  # within limits [-1, 1]
    limits = jnp.array([[-1.0, 1.0], [-1.0, 1.0]])
    energy = jnp.array(0.0)
    assert float(physics_loss(torques, limits, energy)) == pytest.approx(0.0)


def test_physics_loss_positive_on_violation() -> None:
    torques = jnp.array([2.0])  # exceeds max of 1.0
    limits = jnp.array([[-1.0, 1.0]])
    energy = jnp.array(0.0)
    assert float(physics_loss(torques, limits, energy)) > 0


def test_physics_loss_violation_amount() -> None:
    torques = jnp.array([3.0])  # exceeds max of 1.0 by 2.0
    limits = jnp.array([[-1.0, 1.0]])
    energy = jnp.array(0.0)
    # violation = 3.0 - 1.0 = 2.0; energy = 0.0
    assert float(physics_loss(torques, limits, energy)) == pytest.approx(2.0)


def test_physics_loss_lower_bound_violation() -> None:
    torques = jnp.array([-3.0])  # below min of -1.0 by 2.0
    limits = jnp.array([[-1.0, 1.0]])
    energy = jnp.array(0.0)
    assert float(physics_loss(torques, limits, energy)) == pytest.approx(2.0)


def test_physics_loss_energy_contribution() -> None:
    torques = jnp.array([0.0])  # within limits
    limits = jnp.array([[-1.0, 1.0]])
    energy = jnp.array(5.0)  # large energy delta
    assert float(physics_loss(torques, limits, energy)) == pytest.approx(5.0)


def test_physics_loss_scalar() -> None:
    """Return value must be scalar."""
    result = physics_loss(
        jnp.array([0.0, 0.0]),
        jnp.array([[-1.0, 1.0], [-1.0, 1.0]]),
        jnp.array(0.0),
    )
    assert result.ndim == 0


# ---------------------------------------------------------------------------
# contact_loss
# ---------------------------------------------------------------------------


def test_contact_loss_zero_for_no_impact() -> None:
    assert float(contact_loss(jnp.zeros(3), stiffness_coeff=10.0)) == pytest.approx(0.0)


def test_contact_loss_positive_for_nonzero_torques() -> None:
    torques = jnp.array([1.0, 2.0])
    result = float(contact_loss(torques, stiffness_coeff=1.0))
    assert result > 0


def test_contact_loss_scaled_by_stiffness() -> None:
    torques = jnp.array([1.0])
    # contact_loss = stiffness_coeff * sum(torques^2) = 2.0 * 1.0 = 2.0
    assert float(contact_loss(torques, stiffness_coeff=2.0)) == pytest.approx(2.0)


def test_contact_loss_scalar() -> None:
    """Return value must be scalar."""
    result = contact_loss(jnp.array([1.0, 0.5]), stiffness_coeff=1.0)
    assert result.ndim == 0


# ---------------------------------------------------------------------------
# total_loss
# ---------------------------------------------------------------------------


def test_total_loss_weighted_sum() -> None:
    weights = LossWeights(data_weight=1.0, physics_weight=0.0001, contact_weight=0.0001)
    pred = jnp.array([1.0])
    actual = jnp.zeros(1)
    torques = jnp.array([0.0])
    limits = jnp.array([[-1.0, 1.0]])
    energy = jnp.array(0.0)
    impact = jnp.zeros(1)
    loss = total_loss(weights, pred, actual, torques, limits, energy, impact, 1.0)
    # With physics and contact both zero (torques within limits, no impact),
    # total = data_weight * data_loss
    expected_data = float(data_loss(pred, actual))
    assert float(loss) == pytest.approx(weights.data_weight * expected_data, rel=1e-5)


def test_total_loss_data_only_weight() -> None:
    """When physics_weight and contact_weight are tiny and inputs are clean,
    total_loss ≈ data_weight * data_loss."""
    weights = LossWeights(data_weight=2.0, physics_weight=1e-9, contact_weight=1e-9)
    pred = jnp.array([1.0])
    actual = jnp.zeros(1)
    torques = jnp.zeros(1)
    limits = jnp.array([[-1.0, 1.0]])
    energy = jnp.array(0.0)
    impact = jnp.zeros(1)
    result = float(
        total_loss(weights, pred, actual, torques, limits, energy, impact, 0.0)
    )
    expected = 2.0 * float(data_loss(pred, actual))
    assert result == pytest.approx(expected, rel=1e-5)


def test_total_loss_positive() -> None:
    weights = LossWeights(data_weight=1.0, physics_weight=1.0, contact_weight=1.0)
    pred = jnp.array([1.0, 2.0])
    actual = jnp.zeros(2)
    torques = jnp.array([5.0, 5.0])
    limits = jnp.array([[-1.0, 1.0], [-1.0, 1.0]])
    energy = jnp.array(3.0)
    impact = jnp.array([1.0, 2.0])
    result = float(
        total_loss(weights, pred, actual, torques, limits, energy, impact, 1.0)
    )
    assert result > 0


def test_total_loss_scalar() -> None:
    """total_loss must return a scalar."""
    weights = LossWeights(data_weight=1.0, physics_weight=1.0, contact_weight=1.0)
    result = total_loss(
        weights,
        jnp.array([1.0]),
        jnp.zeros(1),
        jnp.zeros(1),
        jnp.array([[-1.0, 1.0]]),
        jnp.array(0.0),
        jnp.zeros(1),
        1.0,
    )
    assert result.ndim == 0
