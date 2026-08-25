"""Contracts for the pinned Tools force-attribution gateway."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.biomechanics.force_source_attribution import (
    REQUIRED_FORCE_ATTRIBUTION_SCHEMA,
    attribute_double_pendulum_trajectory,
)
from src.shared.python.simulation_backends import GolfModelParams, make_backend


@pytest.mark.unit
def test_gateway_closes_against_upstream_double_pendulum_backend() -> None:
    params = GolfModelParams.default()
    time = np.linspace(0.0, 0.2, 5)
    q = np.column_stack((0.2 + time, -0.8 + 0.4 * time))
    velocity = np.tile(np.array([3.0, -4.0]), (time.size, 1))
    controls = np.tile(np.array([12.0, -2.0]), (time.size, 1))

    result = attribute_double_pendulum_trajectory(params, time, q, velocity, controls)

    backend = make_backend("ode", params)
    expected = np.stack(
        [
            backend.forward_dynamics(q_row, v_row, u_row)
            for q_row, v_row, u_row in zip(q, velocity, controls, strict=True)
        ]
    )
    np.testing.assert_allclose(result.acceleration_rad_s2, expected, atol=1e-12)
    assert result.schema_version == REQUIRED_FORCE_ATTRIBUTION_SCHEMA
    assert result.endpoint_name == "wrist_hand_path"
    assert result.metrics["coriolis"].signed_tangent_impulse_n_s is not None


@pytest.mark.unit
def test_gateway_retains_velocity_bias_residual_instead_of_hiding_it() -> None:
    params = GolfModelParams.default()
    time = np.array([0.0, 0.1])
    q = np.array([[0.2, -0.7], [0.3, -0.6]])
    velocity = np.array([[2.0, -3.0], [2.5, -3.5]])
    controls = np.zeros((2, 2))

    result = attribute_double_pendulum_trajectory(params, time, q, velocity, controls)

    np.testing.assert_allclose(
        result.components["velocity_residual"].generalized_drive_nm,
        0.0,
        atol=1e-12,
    )
