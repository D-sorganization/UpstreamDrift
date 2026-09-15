"""Conversion bridge from existing UpstreamDrift physics outputs to canonical ShotEnvelope.

Follows TDD, DbC, Law of Demeter, and DRY.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from src.shared.python.golf_simulator.contracts import (
    ClubData,
    ContactStatus,
    NumericalStatus,
    ScientificStatus,
    ShotEnvelope,
    ShotMetadata,
    ShotQualification,
    SourceKind,
)

if TYPE_CHECKING:
    from src.shared.python.physics.ball_launch_conditions import LaunchConditions
    from src.shared.python.physics.impact_model.types import PostImpactState
    from src.shared.python.physics.swing_ball_flight_pipeline import PipelineResult


def rpm_to_rad_s(rpm: float) -> float:
    """Convert revolutions per minute to radians per second.

    Formula: rpm * (2 * pi / 60)
    """
    if not math.isfinite(rpm):
        raise ValueError(f"rpm must be finite, got {rpm!r}")
    return float(rpm) * (2.0 * math.pi / 60.0)


def _resolve_qualification(
    metadata: ShotMetadata, default_source: SourceKind
) -> ShotQualification:
    if metadata.qualification is not None:
        return metadata.qualification
    is_manual = metadata.source_kind == SourceKind.MANUAL
    return ShotQualification(
        contact=ContactStatus.NOT_APPLICABLE if is_manual else ContactStatus.QUALIFIED,
        numerical=NumericalStatus.ESTIMATED if is_manual else NumericalStatus.CONVERGED,
        scientific=ScientificStatus.NOT_APPLICABLE
        if is_manual
        else ScientificStatus.BENCHMARKED,
        evidence_refs=("bridge_conversion",),
    )


def launch_conditions_to_shot_envelope(
    launch_conditions: LaunchConditions,
    metadata: ShotMetadata,
) -> ShotEnvelope:
    """Convert existing LaunchConditions into a canonical ShotEnvelope.

    Decomposes speed, launch angle, and azimuth into Cartesian coordinates (+x forward, +y left, +z up).
    """
    speed = float(launch_conditions.velocity)
    theta = float(launch_conditions.launch_angle)
    phi = float(launch_conditions.azimuth_angle)

    v_horiz = speed * math.cos(theta)
    vx = v_horiz * math.cos(phi)
    vy = v_horiz * math.sin(phi)
    vz = speed * math.sin(theta)

    # Decompose spin:
    omega_mag = rpm_to_rad_s(float(launch_conditions.spin_rate))
    axis = np.asarray(launch_conditions.spin_axis, dtype=float)
    norm = np.linalg.norm(axis)
    unit_axis = axis / norm if norm > 1e-9 else np.array([0.0, -1.0, 0.0])

    angular_vel = (
        float(unit_axis[0] * omega_mag),
        float(unit_axis[1] * omega_mag),
        float(unit_axis[2] * omega_mag),
    )

    qualification = _resolve_qualification(metadata, SourceKind.MANUAL)

    return ShotEnvelope(
        schema_version=1,
        shot_id=metadata.shot_id,
        session_id=metadata.session_id,
        source_kind=metadata.source_kind,
        qualification=qualification,
        ball_velocity_m_s=(vx, vy, vz),
        ball_angular_velocity_rad_s=angular_vel,
        aim_context=metadata.aim_context,
        created_at_utc=metadata.created_at_utc,
        club_data=metadata.club_data,
        model_run_id=metadata.model_run_id,
        trace_digest=metadata.trace_digest,
        impact_id=metadata.impact_id,
        impact_time_s=metadata.impact_time_s,
    )


def post_impact_state_to_shot_envelope(
    post_impact: PostImpactState,
    metadata: ShotMetadata,
) -> ShotEnvelope:
    """Convert PostImpactState from impact model into a canonical ShotEnvelope."""
    vel = tuple(float(x) for x in post_impact.ball_velocity)
    spin = tuple(float(x) for x in post_impact.ball_angular_velocity)

    qualification = _resolve_qualification(metadata, SourceKind.MODEL_CONTACT)

    return ShotEnvelope(
        schema_version=1,
        shot_id=metadata.shot_id,
        session_id=metadata.session_id,
        source_kind=metadata.source_kind,
        qualification=qualification,
        ball_velocity_m_s=(vel[0], vel[1], vel[2]),
        ball_angular_velocity_rad_s=(spin[0], spin[1], spin[2]),
        aim_context=metadata.aim_context,
        created_at_utc=metadata.created_at_utc,
        club_data=metadata.club_data,
        model_run_id=metadata.model_run_id,
        trace_digest=metadata.trace_digest,
        impact_id=metadata.impact_id,
        impact_time_s=metadata.impact_time_s,
    )


def pipeline_result_to_shot_envelope(
    pipeline_result: PipelineResult,
    metadata: ShotMetadata,
) -> ShotEnvelope:
    """Convert PipelineResult into a canonical ShotEnvelope preserving impact state."""
    club_data = metadata.club_data
    if club_data is None and pipeline_result.swing_state is not None:
        ss = pipeline_result.swing_state
        club_data = ClubData(
            club_speed_m_s=float(ss.club_speed) if hasattr(ss, "club_speed") else None,
            attack_angle_rad=float(ss.attack_angle)
            if hasattr(ss, "attack_angle")
            else None,
            club_path_rad=float(ss.club_path) if hasattr(ss, "club_path") else None,
            face_to_target_rad=float(ss.face_angle)
            if hasattr(ss, "face_angle")
            else None,
        )

    updated_metadata = ShotMetadata(
        shot_id=metadata.shot_id,
        session_id=metadata.session_id,
        aim_context=metadata.aim_context,
        created_at_utc=metadata.created_at_utc,
        source_kind=metadata.source_kind,
        qualification=metadata.qualification,
        club_data=club_data,
        model_run_id=metadata.model_run_id,
        trace_digest=metadata.trace_digest,
        impact_id=metadata.impact_id,
        impact_time_s=metadata.impact_time_s,
    )

    return post_impact_state_to_shot_envelope(
        post_impact=pipeline_result.impact_state,
        metadata=updated_metadata,
    )
