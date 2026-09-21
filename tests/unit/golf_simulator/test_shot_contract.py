"""Unit tests for golf simulator contracts and immutable shot envelopes.

Follows TDD, DbC, LoD and DRY principles.
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
import numpy as np
import pytest

from src.shared.python.golf_simulator.contracts import (
    AimContext,
    CapabilityDescriptor,
    CapabilityState,
    ClubData,
    ContactStatus,
    NumericalStatus,
    ScientificStatus,
    ShotEnvelope,
    ShotQualification,
    SimulatorCapabilities,
    SourceKind,
    SUPPORTED_SCHEMA_VERSIONS,
)

pytestmark = pytest.mark.unit


def _valid_aim_context() -> AimContext:
    # 3x3 identity matrix as tuple of tuples
    return AimContext(
        source_to_target_rotation=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        revision=1,
        provenance="default_identity",
    )


def _valid_qualification(
    source_kind: SourceKind = SourceKind.MANUAL,
) -> ShotQualification:
    if source_kind == SourceKind.MANUAL:
        return ShotQualification(
            contact=ContactStatus.NOT_APPLICABLE,
            numerical=NumericalStatus.ESTIMATED,
            scientific=ScientificStatus.NOT_APPLICABLE,
            evidence_refs=("manual_entry",),
        )
    return ShotQualification(
        contact=ContactStatus.QUALIFIED,
        numerical=NumericalStatus.CONVERGED,
        scientific=ScientificStatus.BENCHMARKED,
        evidence_refs=("model_run_123_receipt",),
    )


def test_valid_manual_shot_envelope_creation() -> None:
    envelope = ShotEnvelope(
        schema_version=1,
        shot_id="shot-uuid-001",
        session_id="session-uuid-001",
        source_kind=SourceKind.MANUAL,
        qualification=_valid_qualification(SourceKind.MANUAL),
        ball_velocity_m_s=(70.0, 1.0, 15.0),
        ball_angular_velocity_rad_s=(0.0, -250.0, 10.0),
        aim_context=_valid_aim_context(),
        created_at_utc="2026-09-15T12:00:00Z",
    )
    assert envelope.schema_version == 1
    assert envelope.shot_id == "shot-uuid-001"
    assert envelope.ball_velocity_m_s == (70.0, 1.0, 15.0)
    assert envelope.frame == "target_local_xyz"
    assert envelope.source_kind == SourceKind.MANUAL
    assert envelope.model_run_id is None
    assert envelope.impact_id is None


def test_shot_envelope_immutability() -> None:
    envelope = ShotEnvelope(
        schema_version=1,
        shot_id="shot-uuid-001",
        session_id="session-uuid-001",
        source_kind=SourceKind.MANUAL,
        qualification=_valid_qualification(SourceKind.MANUAL),
        ball_velocity_m_s=(70.0, 0.0, 15.0),
        ball_angular_velocity_rad_s=(0.0, -250.0, 0.0),
        aim_context=_valid_aim_context(),
        created_at_utc="2026-09-15T12:00:00Z",
    )
    with pytest.raises(FrozenInstanceError):
        envelope.ball_velocity_m_s = (80.0, 0.0, 15.0)  # type: ignore[misc]


def test_snapshot_ownership_defensive_copy() -> None:
    arr_vel = np.array([70.0, 0.0, 15.0])
    arr_spin = np.array([0.0, -250.0, 0.0])
    envelope = ShotEnvelope(
        schema_version=1,
        shot_id="shot-uuid-001",
        session_id="session-uuid-001",
        source_kind=SourceKind.MANUAL,
        qualification=_valid_qualification(SourceKind.MANUAL),
        ball_velocity_m_s=tuple(arr_vel),
        ball_angular_velocity_rad_s=tuple(arr_spin),
        aim_context=_valid_aim_context(),
        created_at_utc="2026-09-15T12:00:00Z",
    )
    arr_vel[0] = 999.0
    arr_spin[1] = 999.0
    assert envelope.ball_velocity_m_s == (70.0, 0.0, 15.0)
    assert envelope.ball_angular_velocity_rad_s == (0.0, -250.0, 0.0)


def test_reject_nan_and_inf() -> None:
    with pytest.raises(ValueError, match="finite"):
        ShotEnvelope(
            schema_version=1,
            shot_id="shot-uuid-001",
            session_id="session-uuid-001",
            source_kind=SourceKind.MANUAL,
            qualification=_valid_qualification(SourceKind.MANUAL),
            ball_velocity_m_s=(float("nan"), 0.0, 15.0),
            ball_angular_velocity_rad_s=(0.0, -250.0, 0.0),
            aim_context=_valid_aim_context(),
            created_at_utc="2026-09-15T12:00:00Z",
        )

    with pytest.raises(ValueError, match="finite"):
        ShotEnvelope(
            schema_version=1,
            shot_id="shot-uuid-001",
            session_id="session-uuid-001",
            source_kind=SourceKind.MANUAL,
            qualification=_valid_qualification(SourceKind.MANUAL),
            ball_velocity_m_s=(70.0, float("inf"), 15.0),
            ball_angular_velocity_rad_s=(0.0, -250.0, 0.0),
            aim_context=_valid_aim_context(),
            created_at_utc="2026-09-15T12:00:00Z",
        )


def test_reject_booleans_as_numeric_payloads() -> None:
    with pytest.raises(TypeError, match="bool"):
        ShotEnvelope(
            schema_version=1,
            shot_id="shot-uuid-001",
            session_id="session-uuid-001",
            source_kind=SourceKind.MANUAL,
            qualification=_valid_qualification(SourceKind.MANUAL),
            ball_velocity_m_s=(True, 0.0, 15.0),  # type: ignore[arg-type]
            ball_angular_velocity_rad_s=(0.0, -250.0, 0.0),
            aim_context=_valid_aim_context(),
            created_at_utc="2026-09-15T12:00:00Z",
        )


def test_reject_zero_speed_strike() -> None:
    with pytest.raises(ValueError, match="zero-speed.*no strike"):
        ShotEnvelope(
            schema_version=1,
            shot_id="shot-uuid-001",
            session_id="session-uuid-001",
            source_kind=SourceKind.MANUAL,
            qualification=_valid_qualification(SourceKind.MANUAL),
            ball_velocity_m_s=(0.0, 0.0, 0.0),
            ball_angular_velocity_rad_s=(0.0, 0.0, 0.0),
            aim_context=_valid_aim_context(),
            created_at_utc="2026-09-15T12:00:00Z",
        )


def test_reject_unsupported_schema_version() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        ShotEnvelope(
            schema_version=999,
            shot_id="shot-uuid-001",
            session_id="session-uuid-001",
            source_kind=SourceKind.MANUAL,
            qualification=_valid_qualification(SourceKind.MANUAL),
            ball_velocity_m_s=(70.0, 0.0, 15.0),
            ball_angular_velocity_rad_s=(0.0, -250.0, 0.0),
            aim_context=_valid_aim_context(),
            created_at_utc="2026-09-15T12:00:00Z",
        )


def test_model_source_requires_provenance_and_impact() -> None:
    # Model contact source without model_run_id or trace_digest
    with pytest.raises(ValueError, match="model_run_id and trace_digest"):
        ShotEnvelope(
            schema_version=1,
            shot_id="shot-uuid-001",
            session_id="session-uuid-001",
            source_kind=SourceKind.MODEL_CONTACT,
            qualification=_valid_qualification(SourceKind.MODEL_CONTACT),
            ball_velocity_m_s=(70.0, 0.0, 15.0),
            ball_angular_velocity_rad_s=(0.0, -250.0, 0.0),
            aim_context=_valid_aim_context(),
            created_at_utc="2026-09-15T12:00:00Z",
            model_run_id=None,
            trace_digest=None,
        )

    # Model contact source with negative impact time
    with pytest.raises(ValueError, match="impact_time_s"):
        ShotEnvelope(
            schema_version=1,
            shot_id="shot-uuid-001",
            session_id="session-uuid-001",
            source_kind=SourceKind.MODEL_CONTACT,
            qualification=_valid_qualification(SourceKind.MODEL_CONTACT),
            ball_velocity_m_s=(70.0, 0.0, 15.0),
            ball_angular_velocity_rad_s=(0.0, -250.0, 0.0),
            aim_context=_valid_aim_context(),
            created_at_utc="2026-09-15T12:00:00Z",
            model_run_id="run-1",
            trace_digest="sha256:abc",
            impact_id="impact-1",
            impact_time_s=-0.05,
        )


def test_aim_context_rotation_validation() -> None:
    # Reflection (det = -1) must be rejected
    reflection_matrix = (
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    with pytest.raises(ValueError, match="proper rotation"):
        AimContext(
            source_to_target_rotation=reflection_matrix,
            revision=1,
            provenance="invalid_reflection",
        )

    # Non-orthogonal matrix must be rejected
    sheared_matrix = (
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    with pytest.raises(ValueError, match="proper rotation"):
        AimContext(
            source_to_target_rotation=sheared_matrix,
            revision=1,
            provenance="sheared",
        )


def test_capabilities_distinct_states() -> None:
    caps = SimulatorCapabilities(
        shot_input=CapabilityDescriptor(
            state=CapabilityState.SUPPORTED,
            evidence="Documented Open Connect port 921 API",
        ),
        club_data=CapabilityDescriptor(
            state=CapabilityState.SUPPORTED,
            evidence="Supported in GSPro payload",
        ),
        native_avatar_animation=CapabilityDescriptor(
            state=CapabilityState.UNSUPPORTED,
            evidence="No runtime avatar injection API documented in reviewed Open Connect v1",
        ),
        course_state_feedback=CapabilityDescriptor(
            state=CapabilityState.UNVERIFIED,
            evidence="Not observed in baseline testing",
        ),
    )
    assert caps.shot_input.state == CapabilityState.SUPPORTED
    assert caps.native_avatar_animation.state == CapabilityState.UNSUPPORTED
    assert caps.course_state_feedback.state == CapabilityState.UNVERIFIED
