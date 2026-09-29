"""Contracts for articulated same-state drift/contact attribution (#9151)."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.articulated_drift_contact_attribution import (
    AttributionAdequacy,
    decompose_generalized_dynamics,
    scale_generalized_coordinates,
)


def test_decomposition_closes_acceleration_power_and_mass_metric_shares() -> None:
    matrix = np.diag([2.0, 3.0])
    bias = np.array([5.0, -4.0])
    zero_velocity_bias = np.array([3.0, -2.0])
    contact = np.array([7.0, 1.0])
    active = np.array([-1.0, 2.0])
    velocity = np.array([0.4, -0.2])

    result = decompose_generalized_dynamics(
        mass_matrix=matrix,
        bias_force=bias,
        zero_velocity_bias_force=zero_velocity_bias,
        contact_force=contact,
        active_force=active,
        velocity=velocity,
    )

    expected_forces = np.array([[-3.0, 2.0], [-2.0, 2.0], [7.0, 1.0], [-1.0, 2.0]])
    np.testing.assert_allclose(result.generalized_forces, expected_forces)
    np.testing.assert_allclose(
        result.acceleration_contributions,
        np.array([np.linalg.solve(matrix, force) for force in expected_forces]),
    )
    np.testing.assert_allclose(
        result.total_acceleration,
        np.linalg.solve(matrix, np.sum(expected_forces, axis=0)),
    )
    assert result.acceleration_closure_residual <= 1.0e-15
    assert result.power_closure_residual_w <= 1.0e-15
    assert result.acceleration_share_adequacy is AttributionAdequacy.ADEQUATE
    assert result.power_share_adequacy is AttributionAdequacy.ADEQUATE
    assert sum(result.mass_metric_acceleration_shares) == pytest.approx(1.0)
    assert sum(result.generalized_power_shares) == pytest.approx(1.0)


def test_share_reporting_suppresses_near_zero_denominators() -> None:
    result = decompose_generalized_dynamics(
        mass_matrix=np.eye(2),
        bias_force=np.zeros(2),
        zero_velocity_bias_force=np.zeros(2),
        contact_force=np.zeros(2),
        active_force=np.zeros(2),
        velocity=np.ones(2),
        share_denominator_floor=1.0e-10,
    )

    assert result.acceleration_share_adequacy is AttributionAdequacy.SUPPRESSED
    assert result.power_share_adequacy is AttributionAdequacy.SUPPRESSED
    assert np.all(np.isnan(result.mass_metric_acceleration_shares))
    assert np.all(np.isnan(result.generalized_power_shares))
    assert np.isnan(result.acceleration_cancellation_index)
    assert np.isnan(result.power_cancellation_index)


def test_coordinate_scaling_preserves_physical_power_and_mass_metric_shares() -> None:
    matrix = np.array([[3.0, 0.4], [0.4, 1.7]])
    bias = np.array([2.0, -1.0])
    zero_velocity_bias = np.array([1.5, -0.25])
    contact = np.array([0.75, 1.25])
    active = np.array([-0.1, 0.4])
    velocity = np.array([0.8, -0.3])
    scale = np.array([1000.0, 180.0 / np.pi])

    reference = decompose_generalized_dynamics(
        mass_matrix=matrix,
        bias_force=bias,
        zero_velocity_bias_force=zero_velocity_bias,
        contact_force=contact,
        active_force=active,
        velocity=velocity,
    )
    transformed = scale_generalized_coordinates(
        mass_matrix=matrix,
        bias_force=bias,
        zero_velocity_bias_force=zero_velocity_bias,
        contact_force=contact,
        active_force=active,
        velocity=velocity,
        coordinate_scale=scale,
    )
    scaled = decompose_generalized_dynamics(**transformed)

    np.testing.assert_allclose(
        scaled.generalized_powers_w, reference.generalized_powers_w
    )
    np.testing.assert_allclose(
        scaled.mass_metric_acceleration_shares,
        reference.mass_metric_acceleration_shares,
    )
    np.testing.assert_allclose(
        scaled.generalized_power_shares,
        reference.generalized_power_shares,
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("mass_matrix", np.ones((2, 3)), "square"),
        ("bias_force", np.ones(3), "shape"),
        ("share_denominator_floor", 0.0, "positive"),
    ],
)
def test_decomposition_rejects_invalid_contracts(
    field: str, value: object, message: str
) -> None:
    arguments: dict[str, object] = {
        "mass_matrix": np.eye(2),
        "bias_force": np.zeros(2),
        "zero_velocity_bias_force": np.zeros(2),
        "contact_force": np.zeros(2),
        "active_force": np.zeros(2),
        "velocity": np.zeros(2),
        "share_denominator_floor": 1.0e-12,
    }
    arguments[field] = value
    with pytest.raises(ValueError, match=message):
        decompose_generalized_dynamics(**arguments)  # type: ignore[arg-type]
