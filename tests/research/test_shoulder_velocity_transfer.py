"""Scientific contracts for the shoulder-velocity drift-transfer study."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.shoulder_velocity_transfer import (
    VelocitySweepRequest,
    classify_transfer_phase,
    evaluate_velocity_sweep,
    nondominated_indices,
)
from src.shared.python.simulation_backends import GolfModelParams

pytestmark = pytest.mark.scientific


def _request() -> VelocitySweepRequest:
    return VelocitySweepRequest(
        q_rad=np.array([-0.45, -0.80]),
        reference_velocity_rad_s=np.array([7.0, 5.0]),
        control_nm=np.array([60.0, -4.0]),
        proximal_velocity_rad_s=np.array([3.0, 6.0, 9.0]),
        velocity_constraint="preserve_relative_club_rate",
    )


def test_velocity_sweep_has_exact_pointwise_drift_control_closure() -> None:
    rows = evaluate_velocity_sweep(_request(), GolfModelParams.default())

    assert len(rows) == 3
    assert [row.proximal_velocity_rad_s for row in rows] == [3.0, 6.0, 9.0]
    assert all(row.model_tier == "exact_planar_double_pendulum" for row in rows)
    assert all(abs(row.acceleration_closure_residual_rad_s2) < 1e-10 for row in rows)
    assert all(abs(row.force_closure_residual_n) < 1e-10 for row in rows)
    assert not np.isclose(rows[0].drift_grip_power_w, rows[-1].drift_grip_power_w)


def test_velocity_constraint_can_preserve_absolute_club_rate() -> None:
    request = VelocitySweepRequest(
        q_rad=_request().q_rad,
        reference_velocity_rad_s=np.array([7.0, 5.0]),
        control_nm=_request().control_nm,
        proximal_velocity_rad_s=np.array([3.0, 9.0]),
        velocity_constraint="preserve_absolute_club_rate",
    )

    rows = evaluate_velocity_sweep(request, GolfModelParams.default())

    assert rows[0].club_angular_velocity_rad_s == pytest.approx(12.0)
    assert rows[1].club_angular_velocity_rad_s == pytest.approx(12.0)


def test_request_rejects_duplicate_or_ambiguous_velocity_contract() -> None:
    with pytest.raises(ValueError, match="unique"):
        VelocitySweepRequest(
            q_rad=np.zeros(2),
            reference_velocity_rad_s=np.ones(2),
            control_nm=np.zeros(2),
            proximal_velocity_rad_s=np.array([2.0, 2.0]),
        )
    with pytest.raises(ValueError, match="velocity_constraint"):
        VelocitySweepRequest(
            q_rad=np.zeros(2),
            reference_velocity_rad_s=np.ones(2),
            control_nm=np.zeros(2),
            proximal_velocity_rad_s=np.array([1.0, 2.0]),
            velocity_constraint="hold_everything",
        )


@pytest.mark.parametrize(
    ("fraction", "expected"),
    [
        (0.0, "Transition"),
        (0.15, "Early Downswing"),
        (0.40, "Mid-Downswing"),
        (0.70, "Delivery and Release"),
        (0.95, "Pre-Impact"),
    ],
)
def test_phase_classifier_uses_declared_normalized_boundaries(
    fraction: float, expected: str
) -> None:
    assert classify_transfer_phase(fraction) == expected


def test_nondominated_indices_preserve_speed_braking_tradeoff() -> None:
    objectives = np.array([[20.0, 2.0], [19.0, 1.0], [18.0, 3.0]])
    assert np.array_equal(
        nondominated_indices(objectives, maximize=(True, False)), np.array([0, 1])
    )
