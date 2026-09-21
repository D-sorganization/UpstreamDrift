"""Conversion bridge from existing UpstreamDrift physics outputs to canonical ShotEnvelope.

Follows TDD, DbC, Law of Demeter, and DRY.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

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
    from src.shared.python.physics.impact_model.types import (
        PostImpactState,
        PreImpactState,
    )
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
        scientific=(
            ScientificStatus.NOT_APPLICABLE
            if is_manual
            else ScientificStatus.BENCHMARKED
        ),
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


def shot_envelope_to_launch_conditions(
    shot: ShotEnvelope,
) -> LaunchConditions:
    """Convert a canonical ShotEnvelope back into standard LaunchConditions.

    Recovers speed, vertical launch angle, horizontal azimuth, spin rate (RPM),
    and unit spin axis from Cartesian SI coordinates (+x forward, +y left, +z up).
    """
    from src.shared.python.physics.ball_launch_conditions import LaunchConditions

    vx, vy, vz = shot.ball_velocity_m_s
    v_horiz = math.sqrt(vx * vx + vy * vy)
    speed = math.sqrt(vx * vx + vy * vy + vz * vz)
    launch_angle = math.atan2(vz, v_horiz)
    azimuth_angle = math.atan2(vy, vx)

    wx, wy, wz = shot.ball_angular_velocity_rad_s
    omega_mag = math.sqrt(wx * wx + wy * wy + wz * wz)
    spin_rate_rpm = omega_mag * 60.0 / (2.0 * math.pi)

    if omega_mag > 1e-9:
        spin_axis = np.array(
            [wx / omega_mag, wy / omega_mag, wz / omega_mag], dtype=float
        )
    else:
        spin_axis = np.array([0.0, -1.0, 0.0], dtype=float)

    return LaunchConditions(
        velocity=float(speed),
        launch_angle=float(launch_angle),
        azimuth_angle=float(azimuth_angle),
        spin_rate=float(spin_rate_rpm),
        spin_axis=spin_axis,
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
            attack_angle_rad=(
                float(ss.attack_angle) if hasattr(ss, "attack_angle") else None
            ),
            club_path_rad=float(ss.club_path) if hasattr(ss, "club_path") else None,
            face_to_target_rad=(
                float(ss.face_angle) if hasattr(ss, "face_angle") else None
            ),
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


def qualify_impact_contact(
    pre_state: PreImpactState,
    post_state: PostImpactState,
    source_kind: SourceKind,
    engine_name: str,
) -> ShotQualification:
    """Qualify the contact and physics integrity of an impact event.

    Audits:
    1. Positive approach velocity (club moving toward ball along normal).
    2. Post-impact ball launch speed > 0.
    3. Physically valid smash factor (ball_speed / club_speed <= 1.60 for golf ball collision).
    4. Distinguishes MODEL_CONTACT vs DEMO_PEAK_SPEED vs MANUAL.
    5. Provenance linking with engine identity and smash factor evidence.

    Raises:
        ValueError: If approach velocity <= 0, ball speed <= 0, or smash factor exceeds 1.60.
    """
    v_club = np.asarray(pre_state.clubhead_velocity, dtype=float)
    v_ball_pre = np.asarray(pre_state.ball_velocity, dtype=float)
    n_face = np.asarray(pre_state.clubhead_orientation, dtype=float)
    norm_n = float(np.linalg.norm(n_face))
    n_unit = n_face / norm_n if norm_n > 1e-9 else np.array([1.0, 0.0, 0.0])

    v_rel = v_club - v_ball_pre
    approach_speed = float(np.dot(v_rel, n_unit))
    if approach_speed <= 1e-6:
        raise ValueError(
            f"negative or zero approach velocity: approach_speed={approach_speed:.4f} m/s"
        )

    club_speed = float(np.linalg.norm(v_club))
    v_ball_post = np.asarray(post_state.ball_velocity, dtype=float)
    ball_speed = float(np.linalg.norm(v_ball_post))

    if ball_speed <= 1e-6:
        raise ValueError(
            f"zero or negative post-impact ball speed: {ball_speed:.4f} m/s"
        )

    smash_factor = ball_speed / club_speed if club_speed > 1e-6 else 0.0
    # Theoretical maximum for golf ball (COR ~0.83, mass ratio ~4.3) is ~1.50 - 1.55.
    # We enforce an upper bound of 1.60 to reject unphysical energy injection.
    if smash_factor > 1.60:
        raise ValueError(
            f"Smash factor {smash_factor:.3f} exceeds physical limit 1.60 (ball_speed={ball_speed:.1f}, club_speed={club_speed:.1f})"
        )

    evidence = (
        f"engine:{engine_name}",
        f"smash_factor:{smash_factor:.3f}",
        f"approach_speed_m_s:{approach_speed:.2f}",
    )

    if source_kind == SourceKind.DEMO_PEAK_SPEED:
        return ShotQualification(
            contact=ContactStatus.DEMO_ONLY,
            numerical=NumericalStatus.CONVERGED,
            scientific=ScientificStatus.PEAK_SPEED_HEURISTIC,
            evidence_refs=evidence,
        )

    if source_kind == SourceKind.MODEL_CONTACT:
        return ShotQualification(
            contact=ContactStatus.QUALIFIED,
            numerical=NumericalStatus.CONVERGED,
            scientific=ScientificStatus.BENCHMARKED,
            evidence_refs=evidence,
        )

    return ShotQualification(
        contact=ContactStatus.UNVERIFIED,
        numerical=NumericalStatus.ESTIMATED,
        scientific=ScientificStatus.UNVERIFIED,
        evidence_refs=evidence,
    )


def _make_contact_event(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    peak_sample = max(
        samples,
        key=lambda s: float(s.get("normal_force", 0.0)),
    )
    return {
        "start_time_s": float(samples[0]["time_s"]),
        "end_time_s": float(samples[-1]["time_s"]),
        "peak_time_s": float(peak_sample["time_s"]),
        "max_normal_force": float(peak_sample.get("normal_force", 0.0)),
        "num_samples": len(samples),
    }


def extract_single_contact_event(
    contact_samples: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Identify contiguous contact events and extract peak metrics from each event.

    Deduplicates contiguous in-contact samples into a single contact event with
    identified start, end, peak normal force, and peak timestamp.
    Ensures multiple samples across the contact duration do not produce multiple shots.
    """
    events: list[dict[str, Any]] = []
    in_event = False
    current_samples: list[dict[str, Any]] = []

    for sample in contact_samples:
        is_contact = bool(sample.get("in_contact", False))
        if is_contact:
            in_event = True
            current_samples.append(sample)
        else:
            if in_event and current_samples:
                # Event ended - extract peak
                events.append(_make_contact_event(current_samples))
                current_samples = []
                in_event = False

    if in_event and current_samples:
        events.append(_make_contact_event(current_samples))

    return events


def check_simscape_eligibility(matlab_release: str | None) -> bool:
    """Verify Simscape acceptance eligibility under project policy.

    Requires MATLAB R2025b explicitly.
    """
    if not matlab_release or not isinstance(matlab_release, str):
        return False
    return matlab_release.strip().upper() == "R2025B"
