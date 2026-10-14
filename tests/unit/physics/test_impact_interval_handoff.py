"""Unit and contract tests for pre-impact bundle and contact-interval handoff (#9549, #9703)."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.shared.python.physics._pre_impact_contracts import (
    AbsentFieldError,
    FieldOrigin,
    PreImpactBundleError,
    Quantity,
)
from src.shared.python.physics._pre_impact_frames import (
    Pose,
    shift_twist_reference,
    shift_wrench_origin,
    twist_to_parent,
    wrench_to_parent,
)
from src.shared.python.physics.impact_interval_audit import (
    ImpactAuditError,
    ImpactAuditTolerances,
    recompute_energy_residual,
)
from src.shared.python.physics.impact_interval_handoff import (
    UD_IMPACT_PLAYBACK_SCHEMA,
    UD_IMPACT_PLAYBACK_VERSION,
    UD_PRE_IMPACT_BUNDLE_FORMAT,
    ContactPlaybackFrame,
    create_default_hand_wrenches,
    create_default_modal_shaft,
    create_default_provenance,
    create_default_timebase,
    export_impact_run_playback_summary,
    export_pre_impact_kernel_bundle,
    interpolate_contact_trajectory_frames,
)
from src.shared.python.physics.pre_impact_bundle import (
    PRE_IMPACT_BUNDLE_SCHEMA,
    PRE_IMPACT_BUNDLE_VERSION,
    HandWrench,
    HeadState,
    ModalBasis,
    PreImpactBundle,
    project_onto_basis,
)
from src.shared.python.physics.swing_ball_flight_pipeline import SwingState

pytestmark = pytest.mark.unit

_POST_IMPACT_PAYLOAD = {"status": "valid_post_impact"}


class _MockSample:
    """Mock impact sample returned by result.at_time(t)."""

    def __init__(self, t: float, duration: float) -> None:
        alpha = t / duration if duration > 0 else 0.0
        self.ball_position_m = np.array([alpha * 0.01, 0.0, 0.0])
        self.club_position_m = np.array([alpha * 0.02, 0.0, 0.0])
        self.club_orientation = np.eye(3)
        self.ball_velocity_mps = np.array([alpha * 65.0, alpha * 10.0, 0.0])
        self.club_velocity_mps = np.array([45.0 - alpha * 10.0, 0.0, 0.0])
        self.ball_angular_velocity_rad_s = np.array([0.0, -alpha * 250.0, 0.0])
        self.club_angular_velocity_rad_s = np.array([0.0, 0.0, 5.0])
        # Force profile: parabolic load/unload
        profile = 4.0 * alpha * (1.0 - alpha)
        self.normal_force_n = max(profile * 15000.0, 0.0)
        self.friction_force_n = max(profile * 3000.0, 0.0)
        self.compression_m = max(profile * 0.005, 0.0)
        self.face_angle_deg = 0.5 * alpha
        self.dynamic_loft_deg = 10.5 + 1.0 * alpha
        self.twist_angle_rad = 0.02 * alpha


class _MockIntervalResult:
    """Mock interval solver result tracking inspection channels and audit."""

    def __init__(
        self,
        audit: SimpleNamespace,
        *,
        termination: str = "SEPARATED",
        contact_completed: bool = True,
        duration: float = 0.00045,
        num_samples: int = 100,
        provide_at_time: bool = True,
    ) -> None:
        self.audit = audit
        self.termination = SimpleNamespace(name=termination)
        self.contact_completed = contact_completed
        self.contact_duration_s = duration
        self.time_s = np.linspace(0.0, duration, num_samples)
        self.provide_at_time = provide_at_time
        self.to_post_impact_state_calls = 0

        # Build channels
        samples = [_MockSample(t, duration) for t in self.time_s]
        self._samples = samples
        self.ball_position_m = np.array([s.ball_position_m for s in samples])
        self.club_position_m = np.array([s.club_position_m for s in samples])
        self.club_orientation = np.array([s.club_orientation for s in samples])
        self.ball_velocity_mps = np.array([s.ball_velocity_mps for s in samples])
        self.club_velocity_mps = np.array([s.club_velocity_mps for s in samples])
        self.ball_angular_velocity_rad_s = np.array(
            [s.ball_angular_velocity_rad_s for s in samples]
        )
        self.club_angular_velocity_rad_s = np.array(
            [s.club_angular_velocity_rad_s for s in samples]
        )
        self.normal_force_n = np.array([s.normal_force_n for s in samples])
        self.friction_force_n = np.array([s.friction_force_n for s in samples])
        self.compression_m = np.array([s.compression_m for s in samples])
        self.face_angle_deg = np.array([s.face_angle_deg for s in samples])
        self.dynamic_loft_deg = np.array([s.dynamic_loft_deg for s in samples])
        self.twist_angle_rad = np.array([s.twist_angle_rad for s in samples])

    def at_time(self, t: float) -> _MockSample:
        if not self.provide_at_time:
            raise AttributeError("at_time disabled")
        return _MockSample(t, self.contact_duration_s)

    def channel(self, name: str) -> np.ndarray:
        if hasattr(self, name):
            return getattr(self, name)
        raise ValueError(f"Unknown impact-interval channel: {name}")

    def to_post_impact_state(self) -> Any:
        self.to_post_impact_state_calls += 1
        return _POST_IMPACT_PAYLOAD


def _closed_audit(**overrides: float) -> SimpleNamespace:
    terms: dict[str, float] = {
        "initial_kinetic_energy_j": 250.0,
        "final_kinetic_energy_j": 235.0,
        "dashpot_and_friction_dissipation_j": 13.0,
        "torsional_damping_dissipation_j": 0.5,
        "dissipated_energy_j": 13.5,
        "unilateral_release_energy_j": 0.4,
        "boundary_stored_energy_j": 1.0,
        "stored_contact_energy_initial_j": 0.0,
        "stored_contact_energy_final_j": 0.0,
        "integrated_normal_impulse_n_s": 3.4,
        "integrated_friction_impulse_n_s": 0.0,
        "linear_momentum_residual_n_s": 1.0e-14,
        "supported_momentum_residual_n_m_s": math.nan,
    }
    terms.update(overrides)
    terms.setdefault(
        "energy_residual_j", recompute_energy_residual(SimpleNamespace(**terms))
    )
    return SimpleNamespace(**terms)


# ==============================================================================
# 1. ud.pre_impact_bundle/1 Schema & Serialization Tests (#9703)
# ==============================================================================


def test_export_pre_impact_kernel_bundle_schema_and_roundtrip() -> None:
    bundle_dict = export_pre_impact_kernel_bundle()

    # Check top-level contract requirements
    assert bundle_dict["format"] == UD_PRE_IMPACT_BUNDLE_FORMAT
    assert bundle_dict["schema"] == UD_PRE_IMPACT_BUNDLE_FORMAT
    assert bundle_dict["version"] == 1
    assert bundle_dict["units"] == "SI"

    for required_key in (
        "provenance",
        "timebase",
        "frames",
        "head",
        "ball",
        "shaft",
        "hands",
        "constraints",
    ):
        assert required_key in bundle_dict

    # Check serialization to JSON without error or NaN
    raw_json = json.dumps(bundle_dict)
    assert raw_json is not None

    # Check reloading into PreImpactBundle
    bundle = PreImpactBundle.from_dict(bundle_dict)
    assert isinstance(bundle, PreImpactBundle)
    assert bundle.schema in (PRE_IMPACT_BUNDLE_SCHEMA, UD_PRE_IMPACT_BUNDLE_FORMAT)
    assert bundle.version == PRE_IMPACT_BUNDLE_VERSION

    # Check re-export from loaded bundle
    reexported = export_pre_impact_kernel_bundle(bundle=bundle)
    assert reexported["format"] == UD_PRE_IMPACT_BUNDLE_FORMAT
    assert reexported["version"] == 1


def test_export_from_swing_state_maps_kinematics() -> None:
    swing = SwingState(
        clubhead_velocity=np.array([48.0, -1.0, 2.0]),
        clubhead_angular_velocity=np.array([2.0, -8.0, 35.0]),
        clubhead_orientation=np.array([1.0, 0.0, 0.0]),
        clubhead_mass=0.205,
        clubhead_loft_deg=9.5,
        clubhead_moi=4.8e-4,
        impact_offset=np.array([0.015, -0.005]),
        engine_name="mujoco",
    )

    bundle_dict = export_pre_impact_kernel_bundle(swing=swing)
    head = bundle_dict["head"]

    assert head["mass_kg"]["value"] == pytest.approx(0.205)
    assert head["linear_velocity_mps"]["value"] == [48.0, -1.0, 2.0]
    assert head["angular_velocity_rad_s"]["value"] == [2.0, -8.0, 35.0]
    assert head["contact_offset_m"]["value"] == [0.0, 0.015, -0.005]

    # Reconstruct bundle and verify head tools contact body fields
    bundle = PreImpactBundle.from_dict(bundle_dict)
    body_fields = bundle.head.tools_contact_body_fields()
    assert body_fields["mass_kg"] == pytest.approx(0.205)
    assert body_fields["contact_offset_m"] == (0.0, 0.015, -0.005)


def test_unsupported_schema_rejected() -> None:
    bundle_dict = export_pre_impact_kernel_bundle()
    bundle_dict["schema"] = "invalid.bundle/9"
    with pytest.raises(PreImpactBundleError, match="unsupported_schema"):
        PreImpactBundle.from_dict(bundle_dict)


# ==============================================================================
# 2. Wrench and Elastic State Handoff Tests (#9703)
# ==============================================================================


def test_hand_wrench_power_invariance() -> None:
    bundle_dict = export_pre_impact_kernel_bundle()
    bundle = PreImpactBundle.from_dict(bundle_dict)

    # Power test: Wrench and twist transformed between frames must have invariant power
    force = np.array([25.0, -40.0, 150.0])
    moment = np.array([3.0, 1.2, -4.5])
    linear = np.array([2.0, -1.5, 30.0])
    angular = np.array([1.0, 4.0, -2.0])

    power_ref = float(np.dot(force, linear) + np.dot(moment, angular))

    for frame_a in ("world", "head", "grip"):
        for frame_b in ("world", "head", "grip"):
            pose = bundle.pose_between(frame_b, frame_a)
            f_b, m_b = wrench_to_parent(pose, force, moment)
            v_b, w_b = twist_to_parent(pose, linear, angular)
            power_trans = float(np.dot(f_b, v_b) + np.dot(m_b, w_b))
            assert math.isclose(power_ref, power_trans, rel_tol=1e-12, abs_tol=1e-12)


def test_hand_wrench_origin_shift_invariance() -> None:
    force = np.array([50.0, -20.0, 80.0])
    moment = np.array([2.0, 0.5, -1.8])
    linear = np.array([10.0, 2.0, -1.0])
    angular = np.array([3.0, -2.0, 15.0])
    old_orig = np.array([0.0, 0.0, 0.05])
    new_orig = np.array([0.02, -0.01, 0.10])

    f_shift, m_shift = shift_wrench_origin(force, moment, old_orig, new_orig)
    v_shift, w_shift = shift_twist_reference(linear, angular, old_orig, new_orig)

    p_orig = float(np.dot(force, linear) + np.dot(moment, angular))
    p_shift = float(np.dot(f_shift, v_shift) + np.dot(m_shift, w_shift))
    assert math.isclose(p_orig, p_shift, rel_tol=1e-12, abs_tol=1e-12)


def test_absent_trail_wrench_raises_on_numeric_access() -> None:
    bundle_dict = export_pre_impact_kernel_bundle()
    bundle = PreImpactBundle.from_dict(bundle_dict)

    trail = bundle.hand("trail")
    assert trail.force_n.is_absent
    assert trail.moment_n_m.is_absent

    with pytest.raises(AbsentFieldError):
        _ = trail.force_n.value
    with pytest.raises(AbsentFieldError):
        bundle.hand_wrench_in("trail", "world")


def test_modal_elastic_state_energy_and_projection() -> None:
    # 2-DOF shaft
    shaft = create_default_modal_shaft(
        amplitudes=[0.02, -0.005],
        velocities=[0.4, 0.08],
    )
    bundle_dict = export_pre_impact_kernel_bundle(shaft=shaft)
    bundle = PreImpactBundle.from_dict(bundle_dict)

    # Known energy: 0.5 * qd^T M qd + 0.5 * q^T K q
    # M = eye(2), K = diag([900, 25000])
    # 0.5*(0.4^2 + 0.08^2) + 0.5*(900*0.02^2 + 25000*(-0.005)^2)
    expected_ke = 0.5 * (0.4**2 + 0.08**2)
    expected_pe = 0.5 * (900.0 * 0.02**2 + 25000.0 * (-0.005) ** 2)
    expected_energy = expected_ke + expected_pe

    assert math.isclose(bundle.shaft.modal_energy_j(), expected_energy, rel_tol=1e-12)


# ==============================================================================
# 3. Contact Playback Frame Interpolation Tests (#9549, #9700)
# ==============================================================================


def test_interpolate_contact_trajectory_frames_with_at_time() -> None:
    audit = _closed_audit()
    result = _MockIntervalResult(audit, provide_at_time=True, duration=0.00040)

    frames = interpolate_contact_trajectory_frames(result, num_frames=30)
    assert len(frames) == 30

    # Test monotonic timebase inside [0, 0.00040]
    times = [f.time_s for f in frames]
    assert times[0] == pytest.approx(0.0)
    assert times[-1] == pytest.approx(0.00040)
    assert all(times[i] < times[i + 1] for i in range(len(times) - 1))

    # Mid contact physical checks: compression > 0, normal force > 0
    mid = frames[15]
    assert mid.compression_m > 0.0
    assert mid.normal_force_n > 0.0
    assert len(mid.club_orientation) == 3

    # Ball launches downrange with positive speed
    assert frames[-1].ball_velocity_mps[0] > frames[0].ball_velocity_mps[0]

    # Serialization
    frame_dict = mid.to_dict()
    assert frame_dict["time_s"] == mid.time_s
    assert frame_dict["normal_force_n"] == mid.normal_force_n
    assert len(frame_dict["club_orientation"]) == 3


def test_interpolate_contact_trajectory_frames_array_fallback() -> None:
    audit = _closed_audit()
    result = _MockIntervalResult(audit, provide_at_time=False, duration=0.00050)

    frames = interpolate_contact_trajectory_frames(result, num_frames=20)
    assert len(frames) == 20
    assert frames[0].time_s == pytest.approx(0.0)
    assert frames[-1].time_s == pytest.approx(0.00050)
    assert frames[10].compression_m > 0.0


def test_interpolate_invalid_frame_count_raises() -> None:
    audit = _closed_audit()
    result = _MockIntervalResult(audit)
    with pytest.raises(ValueError, match="num_frames must be >= 2"):
        interpolate_contact_trajectory_frames(result, num_frames=1)


# ==============================================================================
# 4. Live Run / Playback / Export Summary Integration (#9549, #9700)
# ==============================================================================


def test_export_impact_run_playback_summary_qualified() -> None:
    audit = _closed_audit()
    result = _MockIntervalResult(audit, termination="SEPARATED", contact_completed=True)

    summary = export_impact_run_playback_summary(result, num_playback_frames=25)

    assert summary["format"] == UD_IMPACT_PLAYBACK_SCHEMA
    assert summary["version"] == UD_IMPACT_PLAYBACK_VERSION
    assert summary["qualified"] is True
    assert summary["contact_completed"] is True
    assert summary["termination"] == "SEPARATED"
    assert summary["completion_status"] == "separated"
    assert summary["contact_window_s"] == [
        0.0,
        pytest.approx(result.contact_duration_s),
    ]
    assert summary["contact_duration_s"] == pytest.approx(result.contact_duration_s)
    assert summary["frame_count"] == 25
    assert len(summary["playback_frames"]) == 25
    assert summary["post_impact_state"] is _POST_IMPACT_PAYLOAD
    assert summary["audit"]["qualified"] is True

    # JSON round-trip
    json_str = json.dumps(summary)
    assert json_str is not None


# ==============================================================================
# 5. Fail-Closed Rejection of Unqualified Runs (#9549, #9700)
# ==============================================================================


def test_truncated_contact_time_limit_fails_closed() -> None:
    audit = _closed_audit(
        final_kinetic_energy_j=210.0, stored_contact_energy_final_j=8.5
    )
    result = _MockIntervalResult(
        audit,
        termination="TIME_LIMIT",
        contact_completed=False,
    )

    # Frame interpolation directly raises
    with pytest.raises(ImpactAuditError, match="did not separate"):
        interpolate_contact_trajectory_frames(result)

    # Non-strict summary returns fail-closed payload
    summary = export_impact_run_playback_summary(result, strict=False)
    assert summary["qualified"] is False
    assert summary["contact_completed"] is False
    assert summary["termination"] == "TIME_LIMIT"
    assert summary["completion_status"] == "time_limit"
    assert summary["contact_window_s"] is None
    assert summary["playback_frames"] is None
    assert summary["post_impact_state"] is None
    assert result.to_post_impact_state_calls == 0

    # Strict summary raises
    with pytest.raises(ImpactAuditError):
        export_impact_run_playback_summary(result, strict=True)


def test_missed_contact_no_contact_fails_closed() -> None:
    audit = _closed_audit()
    result = _MockIntervalResult(
        audit,
        termination="NO_CONTACT",
        contact_completed=False,
    )

    with pytest.raises(ImpactAuditError):
        interpolate_contact_trajectory_frames(result)

    summary = export_impact_run_playback_summary(result, strict=False)
    assert summary["qualified"] is False
    assert summary["contact_completed"] is False
    assert summary["termination"] == "NO_CONTACT"
    assert summary["completion_status"] == "no_contact"
    assert summary["playback_frames"] is None
    assert summary["post_impact_state"] is None
    assert result.to_post_impact_state_calls == 0


def test_residual_mismatch_fails_closed() -> None:
    audit = _closed_audit(unilateral_release_energy_j=4.2, energy_residual_j=0.0)
    result = _MockIntervalResult(
        audit,
        termination="SEPARATED",
        contact_completed=True,
    )

    with pytest.raises(
        ImpactAuditError, match="does not equal the residual recomputed"
    ):
        interpolate_contact_trajectory_frames(result)

    summary = export_impact_run_playback_summary(result, strict=False)
    assert summary["qualified"] is False
    assert summary["playback_frames"] is None
    assert summary["post_impact_state"] is None
    assert result.to_post_impact_state_calls == 0
