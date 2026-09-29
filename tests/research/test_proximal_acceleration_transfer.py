"""Scientific contracts for pointwise proximal-acceleration interventions."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.proximal_acceleration_transfer import (
    AccelerationSweepRequest,
    evaluate_acceleration_sweep,
)
from src.shared.python.simulation_backends import GolfModelParams

pytestmark = pytest.mark.scientific


def _request() -> AccelerationSweepRequest:
    return AccelerationSweepRequest(
        q_rad=np.array([-0.45, -0.80]),
        velocity_rad_s=np.array([7.0, 5.0]),
        reference_control_nm=np.array([60.0, -4.0]),
        proximal_acceleration_rad_s2=np.array([-20.0, 0.0, 20.0]),
    )


def test_acceleration_sweep_hits_target_and_preserves_state_and_distal_torque() -> None:
    rows = evaluate_acceleration_sweep(_request(), GolfModelParams.default())

    assert len(rows) == 3
    assert [row.proximal_acceleration_rad_s2 for row in rows] == [-20.0, 0.0, 20.0]
    assert all(abs(row.proximal_acceleration_residual_rad_s2) < 1e-10 for row in rows)
    assert all(row.distal_control_nm == pytest.approx(-4.0) for row in rows)
    assert len({round(row.proximal_control_nm, 8) for row in rows}) == 3
    assert all(
        row.total_kinetic_energy_j == pytest.approx(rows[0].total_kinetic_energy_j)
        for row in rows
    )


def test_acceleration_sweep_retains_drift_control_and_force_closure() -> None:
    rows = evaluate_acceleration_sweep(_request(), GolfModelParams.default())

    assert max(abs(row.acceleration_closure_residual_rad_s2) for row in rows) < 1e-10
    assert max(abs(row.force_closure_residual_n) for row in rows) < 1e-10
    assert not np.isclose(rows[0].total_grip_power_w, rows[-1].total_grip_power_w)


def test_acceleration_request_rejects_duplicate_targets() -> None:
    with pytest.raises(ValueError, match="unique"):
        AccelerationSweepRequest(
            q_rad=np.zeros(2),
            velocity_rad_s=np.ones(2),
            reference_control_nm=np.zeros(2),
            proximal_acceleration_rad_s2=np.array([1.0, 1.0]),
        )
