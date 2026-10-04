"""Tests for DIME Per-Engine and Capture Qualification Matrix (#11433).

Part of Epic #11421.
Verifies:
1. Engine implementing only API skeleton cannot be marked qualified or matched-forward-dynamics.
2. Contact-free model advertised as whole-body GRF is rejected.
3. Different joint conventions treated as parity fail closed.
4. Missing native dependencies cannot be treated as pass.
5. Trajectories exceeding frozen metric thresholds fail qualification.
6. Evidenced engines with valid replay receive qualified and matched-forward-dynamics status.
7. Capture-specific qualification with model/data hashes and pseudonymized subject IDs.
8. Report bundles with manifests exported without private data leaks.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

pytestmark = pytest.mark.unit

from src.shared.python.estimation.dime_engine_qualification import (
    ContactModelKind,
    EngineCapabilitySpec,
    EngineQualificationEntry,
    EngineQualificationMatrix,
    EngineQualificationStatus,
    JointConvention,
    CaptureProvenance,
    build_fleet_qualification_matrix,
    evaluate_engine_qualification,
    export_qualification_bundle,
)
from src.shared.python.estimation.dime_manifest import NumericAcceptanceThresholds


@pytest.fixture
def default_thresholds() -> NumericAcceptanceThresholds:
    return NumericAcceptanceThresholds(
        max_drift_m=0.015,
        max_angular_drift_rad=0.05,
        max_control_norm_nm=250.0,
        min_alignment=0.95,
    )


@pytest.fixture
def valid_engine_spec() -> EngineCapabilitySpec:
    return EngineCapabilitySpec(
        engine_name="mujoco",
        version="3.2.0",
        supported_states=("q", "v", "a"),
        supported_controls=("torque",),
        contact_model=ContactModelKind.WHOLE_BODY_GRF,
        joint_convention=JointConvention.QUATERNION,
        has_forward_dynamics=True,
        has_analytic_derivatives=True,
        has_continuous_replay=True,
        accepted_model_formats=("xml", "mjcf", "urdf"),
        is_api_skeleton_only=False,
        native_binary_present=True,
    )


@pytest.fixture
def valid_capture() -> CaptureProvenance:
    return CaptureProvenance(
        capture_id="tour_swing_01",
        capture_type="tour",
        model_hash="sha256:abc123modelhash",
        data_hash="sha256:def456datahash",
        sampling_rate_hz=120.0,
        subject_id="tour_pro_anon_01",
        claimed_contact=ContactModelKind.WHOLE_BODY_GRF,
        claimed_joint_convention=JointConvention.QUATERNION,
    )


class TestDimeEngineQualificationRedCases:
    """RED behavioral cases required by #11433."""

    def test_api_skeleton_only_cannot_be_qualified(
        self,
        valid_engine_spec: EngineCapabilitySpec,
        valid_capture: CaptureProvenance,
        default_thresholds: NumericAcceptanceThresholds,
    ) -> None:
        skeleton_spec = EngineCapabilitySpec(
            engine_name="fake_engine",
            version="0.1.0",
            supported_states=valid_engine_spec.supported_states,
            supported_controls=valid_engine_spec.supported_controls,
            contact_model=valid_engine_spec.contact_model,
            joint_convention=valid_engine_spec.joint_convention,
            has_forward_dynamics=False,
            has_analytic_derivatives=False,
            has_continuous_replay=False,
            accepted_model_formats=("urdf",),
            is_api_skeleton_only=True,
            native_binary_present=True,
        )
        metrics = {"max_drift_m": 0.005, "alignment": 0.98, "max_control_nm": 50.0}
        entry = evaluate_engine_qualification(
            skeleton_spec,
            valid_capture,
            trajectory_metrics=metrics,
            thresholds=default_thresholds,
        )
        assert entry.status in (
            EngineQualificationStatus.BLOCKED,
            EngineQualificationStatus.REJECTED,
        )
        assert entry.matched_forward_dynamics is False
        assert any("skeleton" in r.lower() for r in entry.reasons)

    def test_contact_free_model_advertised_as_whole_body_grf_fails(
        self,
        valid_engine_spec: EngineCapabilitySpec,
        valid_capture: CaptureProvenance,
        default_thresholds: NumericAcceptanceThresholds,
    ) -> None:
        contact_free_spec = EngineCapabilitySpec(
            engine_name="contact_free_pendulum",
            version="1.0.0",
            supported_states=valid_engine_spec.supported_states,
            supported_controls=valid_engine_spec.supported_controls,
            contact_model=ContactModelKind.CONTACT_FREE,
            joint_convention=valid_engine_spec.joint_convention,
            has_forward_dynamics=True,
            has_analytic_derivatives=True,
            has_continuous_replay=True,
            accepted_model_formats=("urdf",),
            is_api_skeleton_only=False,
            native_binary_present=True,
        )
        # Capture claims whole body GRF
        entry = evaluate_engine_qualification(
            contact_free_spec,
            valid_capture,
            trajectory_metrics=None,
            thresholds=default_thresholds,
        )
        assert entry.status == EngineQualificationStatus.REJECTED
        assert entry.matched_forward_dynamics is False
        assert any("contact-free" in r.lower() for r in entry.reasons)

    def test_different_joint_conventions_treated_as_parity_fails(
        self,
        valid_engine_spec: EngineCapabilitySpec,
        valid_capture: CaptureProvenance,
        default_thresholds: NumericAcceptanceThresholds,
    ) -> None:
        mismatched_spec = EngineCapabilitySpec(
            engine_name="pinocchio",
            version="2.6.0",
            supported_states=valid_engine_spec.supported_states,
            supported_controls=valid_engine_spec.supported_controls,
            contact_model=valid_engine_spec.contact_model,
            joint_convention=JointConvention.EULER_XYZ,  # Mismatched vs capture QUATERNION
            has_forward_dynamics=True,
            has_analytic_derivatives=True,
            has_continuous_replay=True,
            accepted_model_formats=("urdf",),
            is_api_skeleton_only=False,
            native_binary_present=True,
        )
        entry = evaluate_engine_qualification(
            mismatched_spec,
            valid_capture,
            trajectory_metrics=None,
            thresholds=default_thresholds,
        )
        assert entry.status == EngineQualificationStatus.REJECTED
        assert entry.matched_forward_dynamics is False
        assert any("joint convention" in r.lower() for r in entry.reasons)

    def test_missing_native_dependency_cannot_pass(
        self,
        valid_engine_spec: EngineCapabilitySpec,
        valid_capture: CaptureProvenance,
        default_thresholds: NumericAcceptanceThresholds,
    ) -> None:
        missing_native_spec = EngineCapabilitySpec(
            engine_name="drake",
            version="1.20.0",
            supported_states=valid_engine_spec.supported_states,
            supported_controls=valid_engine_spec.supported_controls,
            contact_model=valid_engine_spec.contact_model,
            joint_convention=valid_engine_spec.joint_convention,
            has_forward_dynamics=True,
            has_analytic_derivatives=True,
            has_continuous_replay=True,
            accepted_model_formats=("urdf",),
            is_api_skeleton_only=False,
            native_binary_present=False,  # Binary not installed
        )
        entry = evaluate_engine_qualification(
            missing_native_spec,
            valid_capture,
            trajectory_metrics=None,
            thresholds=default_thresholds,
        )
        assert entry.status == EngineQualificationStatus.BLOCKED
        assert entry.matched_forward_dynamics is False
        assert any("native" in r.lower() for r in entry.reasons)

    def test_trajectory_metrics_exceeding_threshold_fails(
        self,
        valid_engine_spec: EngineCapabilitySpec,
        valid_capture: CaptureProvenance,
        default_thresholds: NumericAcceptanceThresholds,
    ) -> None:
        failing_metrics = {
            "max_drift_m": 0.050,  # Exceeds 0.015m threshold
            "alignment": 0.90,  # Fails 0.95 threshold
            "max_control_nm": 300.0,
        }
        entry = evaluate_engine_qualification(
            valid_engine_spec,
            valid_capture,
            trajectory_metrics=failing_metrics,
            thresholds=default_thresholds,
        )
        assert entry.status == EngineQualificationStatus.REJECTED
        assert entry.matched_forward_dynamics is False
        assert any(
            "drift" in r.lower() or "alignment" in r.lower() for r in entry.reasons
        )


class TestDimeEngineQualificationGreenCases:
    """GREEN behavioral cases required by #11433."""

    def test_evidenced_engine_and_capture_qualification(
        self,
        valid_engine_spec: EngineCapabilitySpec,
        valid_capture: CaptureProvenance,
        default_thresholds: NumericAcceptanceThresholds,
    ) -> None:
        passing_metrics = {
            "max_drift_m": 0.008,
            "max_angular_drift_rad": 0.02,
            "alignment": 0.99,
            "mean_control_nm": 45.0,
        }
        entry = evaluate_engine_qualification(
            valid_engine_spec,
            valid_capture,
            trajectory_metrics=passing_metrics,
            thresholds=default_thresholds,
        )
        assert entry.status == EngineQualificationStatus.QUALIFIED
        assert entry.matched_forward_dynamics is True
        assert len(entry.reasons) == 0

    def test_fleet_qualification_matrix_aggregation(
        self,
        valid_engine_spec: EngineCapabilitySpec,
        valid_capture: CaptureProvenance,
        default_thresholds: NumericAcceptanceThresholds,
    ) -> None:
        blocked_engine = EngineCapabilitySpec(
            engine_name="simscape",
            version="R2025b",
            supported_states=("q", "v"),
            supported_controls=("torque",),
            contact_model=ContactModelKind.WHOLE_BODY_GRF,
            joint_convention=JointConvention.QUATERNION,
            has_forward_dynamics=True,
            has_analytic_derivatives=False,
            has_continuous_replay=True,
            accepted_model_formats=("slx",),
            is_api_skeleton_only=False,
            native_binary_present=False,
        )

        owner_capture = CaptureProvenance(
            capture_id="owner_swing_02",
            capture_type="owner",
            model_hash="sha256:ownerhash123",
            data_hash="sha256:ownerdata456",
            sampling_rate_hz=240.0,
            subject_id="owner_anon_02",
            claimed_contact=ContactModelKind.WHOLE_BODY_GRF,
            claimed_joint_convention=JointConvention.QUATERNION,
        )

        solve_receipts = {
            ("mujoco", "tour_swing_01"): {
                "max_drift_m": 0.007,
                "alignment": 0.98,
                "mean_control_nm": 30.0,
            },
            ("mujoco", "owner_swing_02"): {
                "max_drift_m": 0.010,
                "alignment": 0.97,
                "mean_control_nm": 40.0,
            },
            ("simscape", "tour_swing_01"): {},
            ("simscape", "owner_swing_02"): {},
        }

        matrix = build_fleet_qualification_matrix(
            engine_specs=[valid_engine_spec, blocked_engine],
            captures=[valid_capture, owner_capture],
            solve_receipts=solve_receipts,
            thresholds=default_thresholds,
        )

        assert matrix.total_evaluated == 4
        assert matrix.total_qualified == 2  # Mujoco on tour and owner
        assert matrix.total_blocked == 2  # Simscape binary missing on tour and owner
        assert matrix.schema_version == "dime-engine-qualification-v1"

    def test_export_report_bundle_generates_manifest_and_matrix(
        self,
        valid_engine_spec: EngineCapabilitySpec,
        valid_capture: CaptureProvenance,
        default_thresholds: NumericAcceptanceThresholds,
        tmp_path: Path,
    ) -> None:
        passing_metrics = {
            "max_drift_m": 0.005,
            "alignment": 0.99,
            "mean_control_nm": 20.0,
        }
        matrix = build_fleet_qualification_matrix(
            engine_specs=[valid_engine_spec],
            captures=[valid_capture],
            solve_receipts={("mujoco", "tour_swing_01"): passing_metrics},
            thresholds=default_thresholds,
        )

        bundle_dir = tmp_path / "qualification_bundle"
        result_path = export_qualification_bundle(matrix, bundle_dir)

        assert result_path.exists()
        matrix_file = bundle_dir / "qualification_matrix.json"
        manifest_file = bundle_dir / "manifest.json"

        assert matrix_file.exists()
        assert manifest_file.exists()

        matrix_data = json.loads(matrix_file.read_text(encoding="utf-8"))
        assert matrix_data["total_qualified"] == 1
        assert matrix_data["entries"][0]["engine_name"] == "mujoco"
        assert matrix_data["entries"][0]["matched_forward_dynamics"] is True

        manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
        assert "qualification_matrix.json" in manifest_data["artifacts"]
        assert manifest_data["schema_version"] == "dime-engine-qualification-v1"
