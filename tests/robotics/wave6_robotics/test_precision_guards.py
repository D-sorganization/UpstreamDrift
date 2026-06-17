"""Focused precision guards for robotics sensor/planner utilities."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.robotics.locomotion.footstep_planner import FootstepPlanner
from src.robotics.locomotion.gait_types import GaitParameters
from src.robotics.sensing.force_torque_sensor import ForceTorqueSensor
from src.robotics.sensing.imu_sensor import IMUSensor

pytestmark = pytest.mark.unit


def test_normalize_angle_wraps_large_values_without_looping() -> None:
    planner = FootstepPlanner(GaitParameters())

    wrapped = planner._normalize_angle(1_000_000.0 * math.pi + 0.25)

    assert -math.pi <= wrapped <= math.pi
    assert wrapped == pytest.approx(0.25, abs=1e-9)


def test_normalize_angle_preserves_nan_and_rejects_infinite() -> None:
    planner = FootstepPlanner(GaitParameters())

    assert math.isnan(planner._normalize_angle(math.nan))
    with pytest.raises(ValueError, match="finite"):
        planner._normalize_angle(math.inf)


def test_imu_rejects_zero_quaternion_instead_of_dividing_by_zero() -> None:
    imu = IMUSensor()

    with pytest.raises(ValueError, match="Quaternion norm"):
        imu.set_orientation(np.zeros(4))


def test_imu_keeps_near_unit_quaternion_without_eager_renormalization() -> None:
    imu = IMUSensor()
    q = np.array([1.0 + 1e-12, 0.0, 0.0, 0.0])

    imu.set_orientation(q)

    np.testing.assert_allclose(imu.orientation, q)


def test_force_torque_contact_location_uses_squared_threshold_consistently() -> None:
    sensor = ForceTorqueSensor()
    near_cutoff = np.array([1e-7, 0.0, 0.0, 0.0, 1.0, 0.0])
    valid = np.array([1e-5, 0.0, 0.0, 0.0, 1e-5, 0.0])

    assert sensor.estimate_contact_location(near_cutoff) is None
    location = sensor.estimate_contact_location(valid)

    assert location is not None
    np.testing.assert_allclose(location, [0.0, 0.0, 1.0])
