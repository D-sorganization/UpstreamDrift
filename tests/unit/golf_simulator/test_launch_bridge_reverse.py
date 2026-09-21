"""Unit tests for reverse bridge conversion: ShotEnvelope -> LaunchConditions.

Follows TDD, DbC, and DRY.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytestmark = pytest.mark.unit

from src.shared.python.golf_simulator.contracts import (
    AimContext,
    ContactStatus,
    NumericalStatus,
    ScientificStatus,
    ShotEnvelope,
    ShotMetadata,
    ShotQualification,
    SourceKind,
)
from src.shared.python.golf_simulator.launch_bridge import (
    launch_conditions_to_shot_envelope,
    shot_envelope_to_launch_conditions,
)
from src.shared.python.physics.ball_launch_conditions import LaunchConditions


def _make_sample_envelope(
    vx: float = 65.0,
    vy: float = 0.0,
    vz: float = 15.0,
    wx: float = 0.0,
    wy: float = -260.0,
    wz: float = 0.0,
) -> ShotEnvelope:
    identity_rot = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    aim = AimContext(source_to_target_rotation=identity_rot, revision=1)
    qual = ShotQualification(
        contact=ContactStatus.QUALIFIED,
        numerical=NumericalStatus.CONVERGED,
        scientific=ScientificStatus.BENCHMARKED,
        evidence_refs=("unit_test",),
    )
    return ShotEnvelope(
        schema_version=1,
        shot_id="shot-rev-01",
        session_id="session-rev-01",
        source_kind=SourceKind.MANUAL,
        qualification=qual,
        ball_velocity_m_s=(vx, vy, vz),
        ball_angular_velocity_rad_s=(wx, wy, wz),
        aim_context=aim,
        created_at_utc="2026-09-15T12:00:00Z",
    )


def test_shot_envelope_to_launch_conditions_basic() -> None:
    env = _make_sample_envelope(vx=60.0, vy=0.0, vz=20.0, wy=-200.0)
    lc = shot_envelope_to_launch_conditions(env)

    expected_speed = math.sqrt(60.0**2 + 20.0**2)
    expected_launch_angle = math.atan2(20.0, 60.0)
    assert math.isclose(lc.velocity, expected_speed, rel_tol=1e-6)
    assert math.isclose(lc.launch_angle, expected_launch_angle, rel_tol=1e-6)
    assert math.isclose(lc.azimuth_angle, 0.0, abs_tol=1e-6)

    expected_rpm = 200.0 * 60.0 / (2.0 * math.pi)
    assert math.isclose(lc.spin_rate, expected_rpm, rel_tol=1e-5)
    assert np.allclose(lc.spin_axis, np.array([0.0, -1.0, 0.0]), atol=1e-5)


def test_round_trip_parity_launch_conditions() -> None:
    original_lc = LaunchConditions(
        velocity=72.5,
        launch_angle=math.radians(11.2),
        azimuth_angle=math.radians(-1.8),
        spin_rate=2650.0,
        spin_axis=np.array([0.05, -0.99, 0.10]) / np.linalg.norm([0.05, -0.99, 0.10]),
    )
    identity_rot = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    aim = AimContext(source_to_target_rotation=identity_rot, revision=1)
    meta = ShotMetadata(
        shot_id="shot-rt-01",
        session_id="session-rt-01",
        aim_context=aim,
        created_at_utc="2026-09-15T12:00:00Z",
    )
    envelope = launch_conditions_to_shot_envelope(original_lc, meta)
    recovered_lc = shot_envelope_to_launch_conditions(envelope)

    assert math.isclose(recovered_lc.velocity, original_lc.velocity, rel_tol=1e-6)
    assert math.isclose(
        recovered_lc.launch_angle, original_lc.launch_angle, rel_tol=1e-6
    )
    assert math.isclose(
        recovered_lc.azimuth_angle, original_lc.azimuth_angle, rel_tol=1e-6
    )
    assert math.isclose(recovered_lc.spin_rate, original_lc.spin_rate, rel_tol=1e-5)
    assert np.allclose(recovered_lc.spin_axis, original_lc.spin_axis, atol=1e-5)


def test_zero_spin_envelope_defaults_axis() -> None:
    env = _make_sample_envelope(wx=0.0, wy=0.0, wz=0.0)
    lc = shot_envelope_to_launch_conditions(env)
    assert lc.spin_rate == 0.0
    assert np.allclose(lc.spin_axis, np.array([0.0, -1.0, 0.0]))
