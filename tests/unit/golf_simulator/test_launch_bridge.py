"""Unit tests for converting existing physics outputs into canonical ShotEnvelopes.

Follows TDD, DbC, LoD and DRY principles.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

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
    pipeline_result_to_shot_envelope,
    post_impact_state_to_shot_envelope,
    rpm_to_rad_s,
)
from src.shared.python.physics.ball_launch_conditions import LaunchConditions
from src.shared.python.physics.impact_model.types import PostImpactState

pytestmark = pytest.mark.unit


def _identity_aim() -> AimContext:
    return AimContext(
        source_to_target_rotation=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        revision=1,
        provenance="identity",
    )


def test_rpm_to_rad_s_golden() -> None:
    # 60 RPM is exactly 2 * pi rad/s
    assert math.isclose(rpm_to_rad_s(60.0), 2.0 * math.pi, rel_tol=1e-9)
    # 3000 RPM is 100 * pi rad/s
    assert math.isclose(rpm_to_rad_s(3000.0), 100.0 * math.pi, rel_tol=1e-9)
    # 0 RPM is 0.0 rad/s
    assert rpm_to_rad_s(0.0) == 0.0


def test_convert_launch_conditions_to_shot_envelope() -> None:
    speed = 44.704  # 100 mph in m/s
    launch_deg = 10.0
    azimuth_deg = -2.0  # 2 degrees right (negative in +y left frame)
    spin_rpm = 2500.0
    spin_axis = np.array([0.0, -1.0, 0.0])  # Backspin

    lc = LaunchConditions.from_user_units(
        velocity=speed,
        launch_angle_deg=launch_deg,
        azimuth_deg=azimuth_deg,
        spin_rate_rpm=spin_rpm,
        spin_axis=spin_axis,
    )

    metadata = ShotMetadata(
        shot_id="shot-test-01",
        session_id="session-test-01",
        aim_context=_identity_aim(),
        created_at_utc="2026-09-15T12:00:00Z",
    )

    envelope = launch_conditions_to_shot_envelope(
        launch_conditions=lc,
        metadata=metadata,
    )

    # Expected velocity decomposition:
    v_horiz = speed * math.cos(math.radians(launch_deg))
    expected_vx = v_horiz * math.cos(math.radians(azimuth_deg))
    expected_vy = v_horiz * math.sin(math.radians(azimuth_deg))
    expected_vz = speed * math.sin(math.radians(launch_deg))

    assert math.isclose(envelope.ball_velocity_m_s[0], expected_vx, rel_tol=1e-6)
    assert math.isclose(envelope.ball_velocity_m_s[1], expected_vy, rel_tol=1e-6)
    assert math.isclose(envelope.ball_velocity_m_s[2], expected_vz, rel_tol=1e-6)

    # Expected spin:
    expected_omega = spin_rpm * (2.0 * math.pi / 60.0)
    assert math.isclose(envelope.ball_angular_velocity_rad_s[0], 0.0, abs_tol=1e-6)
    assert math.isclose(
        envelope.ball_angular_velocity_rad_s[1], -expected_omega, rel_tol=1e-6
    )
    assert math.isclose(envelope.ball_angular_velocity_rad_s[2], 0.0, abs_tol=1e-6)

    assert envelope.source_kind == SourceKind.MANUAL
    assert envelope.qualification.contact == ContactStatus.NOT_APPLICABLE


def test_convert_post_impact_state_to_shot_envelope() -> None:
    post = PostImpactState(
        ball_velocity=np.array([65.0, -1.5, 12.0]),
        ball_angular_velocity=np.array([5.0, -280.0, 15.0]),
        clubhead_velocity=np.array([35.0, -0.5, 2.0]),
        clubhead_angular_velocity=np.array([0.0, 0.0, 0.0]),
        contact_duration=0.00045,
        energy_transfer=150.0,
        impact_location=np.array([0.002, -0.001]),
    )

    metadata = ShotMetadata(
        shot_id="shot-model-01",
        session_id="session-model-01",
        model_run_id="run-mujoco-01",
        trace_digest="sha256:fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
        impact_id="impact-evt-01",
        impact_time_s=0.235,
        source_kind=SourceKind.MODEL_CONTACT,
        qualification=ShotQualification(
            contact=ContactStatus.QUALIFIED,
            numerical=NumericalStatus.CONVERGED,
            scientific=ScientificStatus.BENCHMARKED,
            evidence_refs=("mujoco_contact_sensor_01",),
        ),
        aim_context=_identity_aim(),
        created_at_utc="2026-09-15T12:00:00Z",
    )

    envelope = post_impact_state_to_shot_envelope(
        post_impact=post,
        metadata=metadata,
    )

    assert envelope.ball_velocity_m_s == (65.0, -1.5, 12.0)
    assert envelope.ball_angular_velocity_rad_s == (5.0, -280.0, 15.0)
    assert envelope.impact_time_s == 0.235
    assert envelope.model_run_id == "run-mujoco-01"
    assert envelope.qualification.contact == ContactStatus.QUALIFIED


def test_bridge_rejects_zero_speed_launch_conditions() -> None:
    lc = LaunchConditions(
        velocity=0.0,
        launch_angle=0.2,
        azimuth_angle=0.0,
        spin_rate=0.0,
        spin_axis=np.array([0.0, -1.0, 0.0]),
    )
    metadata = ShotMetadata(
        shot_id="shot-zero",
        session_id="session-zero",
        aim_context=_identity_aim(),
        created_at_utc="2026-09-15T12:00:00Z",
    )
    with pytest.raises(ValueError, match="zero-speed"):
        launch_conditions_to_shot_envelope(
            launch_conditions=lc,
            metadata=metadata,
        )
