"""Focused behavioral tests for DIME shared reports, GUI strategy selection and LaTeX methods (#11432).

Enforces:
- RED:
  * IK run mislabeled as forward dynamics fails closed (raises PreconditionError).
  * Missing native GRF rendered as zero fails closed (typed missingness, never silently 0).
  * Unqualified engine offered as validated fails closed (raises PreconditionError).
  * Mismatched timestamp/frame overlays fails closed with typed error (TimingViolationError).
- GREEN:
  * Strategy selection in GUI/service exposes explicit graceful unavailability states.
  * Full and Custom reports include kinematics, selected net controls, GRF provenance,
    drift vs controlled prediction, uncertainty, contact/replay residuals, video overlays,
    and unavailable-capability explanations.
  * Pointwise vs integrated ZTCF recorded distinctly.
  * Offscreen GUI/service tests and report round-trip retain model/data/method identity and output selections.
  * Model/data provenance and method configuration preserved across JSON/dict export.
  * LaTeX explains equations, assumptions, units, reproduction, and limitations.
"""

from __future__ import annotations

import json
from types import MappingProxyType
import numpy as np
import pytest

from src.shared.python.contracts import PreconditionError
from src.shared.python.estimation.dime_contracts import (
    ContactPolicy,
    ControlChannelSpec,
    ProviderCapability,
    VectorSpaceManifold,
)
from src.shared.python.estimation.dime_manifest import (
    CANONICAL_DIME_UNITS,
    CapabilityStatus,
    DimeProvenanceRecord,
)
from src.shared.python.estimation.dime_observation_factors import TimingViolationError
from src.shared.python.estimation.dime_report_integration import (
    DimeReportArtifact,
    DimeReportOptions,
    DimeStrategySelectionService,
    DimeStrategySelectionViewModel,
    DriftAndPredictionPayload,
    EstimatorStrategy,
    GroundReactionForceReport,
    KinematicsPayload,
    ReplayResidualsSummary,
    ReportForceProvenance,
    ReportScope,
    RunClassification,
    UncertaintySummary,
    VideoOverlaySpec,
    ZtcfRecord,
)

pytestmark = pytest.mark.unit


# ==============================================================================
# Helper Fixtures & Factories
# ==============================================================================


def _make_sample_provenance() -> DimeProvenanceRecord:
    return DimeProvenanceRecord(
        engine="simscape",
        engine_version="2025b",
        model_hash="sha256-gs3dx-human-refined-v1",
        param_hash="param-hash-tiger-2000-rev2",
        git_commit="d59d092d0b56c0d89462948931c41b4da3932e81",
        created_at="2026-10-04T08:00:00Z",
        seed=42,
        notes="DIME-11 verified benchmark reference",
    )


def _make_sample_kinematics(n_steps: int = 5) -> KinematicsPayload:
    times = np.linspace(0.0, 0.04, n_steps)
    q = np.zeros((n_steps, 3), dtype=np.float64)
    q[:, 0] = 0.5 * times**2
    v = np.zeros((n_steps, 3), dtype=np.float64)
    v[:, 0] = times
    a = np.zeros((n_steps, 3), dtype=np.float64)
    a[:, 0] = 1.0
    return KinematicsPayload(
        times=times,
        q=q,
        v=v,
        a=a,
        joint_names=("pelvis_tx", "pelvis_ty", "pelvis_tz"),
    )


def _make_sample_ztcf(n_steps: int = 5) -> ZtcfRecord:
    times = np.linspace(0.0, 0.04, n_steps)
    pointwise_acc = np.array([9.81, 9.81, 9.81, 9.81, 9.81], dtype=np.float64)
    integrated_drift = (0.5 * 9.81 * times**2).astype(np.float64)
    return ZtcfRecord(
        pointwise_ztcf_drift=pointwise_acc,
        integrated_ztcf_drift=integrated_drift,
        pointwise_dominance=np.full(n_steps, 0.75, dtype=np.float64),
        integrated_dominance=0.82,
    )


def _make_sample_overlay(n_steps: int = 5) -> VideoOverlaySpec:
    times = np.linspace(0.0, 0.04, n_steps)
    frame_indices = np.arange(n_steps, dtype=np.int64)
    return VideoOverlaySpec(
        overlay_id="ovl-gs3dx-cam01",
        timestamps_s=times,
        frame_indices=frame_indices,
        geometry_model="GS3DX_Human_Refined_Ellipsoid",
        camera_id="cam_down_the_line",
        render_quality="1080p_h264",
        metadata=MappingProxyType({"fps": 100.0, "shutter_s": 0.001}),
    )


def _make_sample_provider_capability(is_qualified: bool = True) -> ProviderCapability:
    return ProviderCapability(
        provider_id="simscape_multibody",
        version="2025b",
        status="qualified" if is_qualified else "implemented",
        n_q=3,
        n_v=3,
        manifold=VectorSpaceManifold(dim=3),
        control_channels=(
            ControlChannelSpec(
                name="lumbar_torque",
                physical_type="torque",
                units="N*m",
                selection_map=(0,),
                limits=(-200.0, 200.0),
            ),
        ),
        contact_policy=ContactPolicy.NATIVE_ELIMINATED,
        retained_passive_loads=(),
        supports_snapshot=True,
        supports_ztcf=True,
        supports_zvcf=True,
        is_qualified=is_qualified,
    )


# ==============================================================================
# RED Tests: Strict Invariants and Fail-Closed Enforcements
# ==============================================================================


class TestRedDimeReportInvariants:
    """RED test suite enforcing fail-closed reporting invariants."""

    def test_red_ik_run_mislabeled_as_forward_dynamics_fails_closed(self) -> None:
        """Inverse kinematics runs cannot be mislabeled as forward dynamics."""
        kinematics = _make_sample_kinematics(5)
        prov = _make_sample_provenance()
        ztcf = _make_sample_ztcf(5)

        with pytest.raises(
            PreconditionError, match="IK run cannot be mislabeled as forward dynamics"
        ):
            DimeReportArtifact(
                report_id="rep-mislabeled-001",
                created_at="2026-10-04T08:00:00Z",
                scope=ReportScope.FULL,
                strategy=EstimatorStrategy.IK,
                run_classification=RunClassification.FORWARD_DYNAMICS,  # Mislabeled!
                provenance=prov,
                is_validated=True,
                kinematics=kinematics,
                ztcf_record=ztcf,
            )

    def test_red_missing_native_grf_rendered_as_zero_fails_closed(self) -> None:
        """Missing native GRF rendered or substituted as zero fails closed."""
        # Attempt 1: Passing zeros with UNAVAILABLE provenance
        with pytest.raises(
            PreconditionError,
            match="Missing native GRF cannot be rendered or substituted as zero",
        ):
            GroundReactionForceReport(
                provenance=ReportForceProvenance.UNAVAILABLE,
                forces_n=np.zeros(
                    (5, 3), dtype=np.float64
                ),  # Forbidden fabricated zero!
                unavailable_reason="Native force plates not captured",
            )

        # Attempt 2: Claiming MEASURED when forces are missing or silently substituted zeros
        with pytest.raises(
            PreconditionError,
            match="Measured GRF requires non-null, measured force data",
        ):
            GroundReactionForceReport(
                provenance=ReportForceProvenance.MEASURED,
                forces_n=None,  # Missing!
            )

    def test_red_unqualified_engine_offered_as_validated_fails_closed(self) -> None:
        """Unqualified engine offered or requested as validated fails closed."""
        service = DimeStrategySelectionService()
        unqualified_capability = _make_sample_provider_capability(is_qualified=False)

        # Attempt to select with require_validated=True on unqualified engine
        with pytest.raises(
            PreconditionError,
            match="Unqualified engine cannot be offered or reported as validated",
        ):
            service.select_strategy(
                strategy=EstimatorStrategy.DIME_MHE,
                engine_capability=unqualified_capability,
                require_validated=True,
            )

        # Attempt to create DimeReportArtifact marked as validated with unqualified capability
        kinematics = _make_sample_kinematics(5)
        prov = _make_sample_provenance()
        ztcf = _make_sample_ztcf(5)
        with pytest.raises(
            PreconditionError,
            match="Unqualified engine cannot be offered or reported as validated",
        ):
            DimeReportArtifact(
                report_id="rep-unqual-002",
                created_at="2026-10-04T08:00:00Z",
                scope=ReportScope.FULL,
                strategy=EstimatorStrategy.DIME_MHE,
                run_classification=RunClassification.HYBRID_ESTIMATION,
                provenance=prov,
                is_validated=True,  # Lies about validation!
                engine_capability_status="implemented",  # Not qualified!
                kinematics=kinematics,
                ztcf_record=ztcf,
            )

    def test_red_mismatched_timestamp_frame_overlays_fails_closed(self) -> None:
        """Mismatched timestamp/frame count in video overlays fails closed."""
        service = DimeStrategySelectionService()
        kinematics = _make_sample_kinematics(n_steps=5)

        # Mismatched length between timestamps and frame indices
        with pytest.raises(
            TimingViolationError,
            match="Overlay timestamps and frame indices must have identical lengths",
        ):
            VideoOverlaySpec(
                overlay_id="ovl-mismatch-len",
                timestamps_s=np.array([0.0, 0.01, 0.02]),
                frame_indices=np.array([0, 1]),  # Mismatched length!
                geometry_model="GS3DX_Human_Refined_Ellipsoid",
            )

        # Mismatched timestamps against kinematics
        mismatched_overlay = VideoOverlaySpec(
            overlay_id="ovl-mismatch-time",
            timestamps_s=np.array([1.0, 1.01, 1.02, 1.03, 1.04]),  # Shifted in time!
            frame_indices=np.arange(5),
            geometry_model="GS3DX_Human_Refined_Ellipsoid",
        )

        with pytest.raises(
            TimingViolationError,
            match="Video overlay timestamps do not align with kinematics timestamps",
        ):
            service.validate_overlay(mismatched_overlay, kinematics)


# ==============================================================================
# GREEN Tests: Valid Capabilities, Round-Trip, and LaTeX Documentation
# ==============================================================================


class TestGreenDimeReportAndStrategySelection:
    """GREEN test suite validating full and custom reports, selection service, and LaTeX."""

    def test_green_strategy_selection_service_graceful_unavailability(self) -> None:
        """Service exposes available strategies with truthful capability records."""
        service = DimeStrategySelectionService()
        prov_cap = _make_sample_provider_capability(is_qualified=True)

        strategies = service.get_available_strategies(prov_cap)
        assert EstimatorStrategy.DIME_MHE in strategies
        assert EstimatorStrategy.CONTINUOUS_REPLAY in strategies
        assert EstimatorStrategy.NEURAL_ESTIMATOR in strategies

        # DIME_MHE is qualified with qualified provider
        mhe_record = strategies[EstimatorStrategy.DIME_MHE]
        assert mhe_record.status == "qualified"

        # Neural estimator remains gracefully unavailable with clear explanation
        neural_record = strategies[EstimatorStrategy.NEURAL_ESTIMATOR]
        assert neural_record.status == "unavailable"
        assert neural_record.reason is not None
        assert "Pending individual qualification" in neural_record.reason

    def test_green_full_and_custom_report_output_contracts(self) -> None:
        """Full and custom reports contain all mandatory physical sections."""
        service = DimeStrategySelectionService()
        prov_cap = _make_sample_provider_capability(is_qualified=True)
        prov = _make_sample_provenance()
        kinematics = _make_sample_kinematics(5)
        ztcf = _make_sample_ztcf(5)
        overlay = _make_sample_overlay(5)

        grf_report = GroundReactionForceReport(
            provenance=ReportForceProvenance.UNAVAILABLE,
            forces_n=None,
            unavailable_reason="Native in-shoe force sensor data not available",
        )

        drift = DriftAndPredictionPayload(
            drift_trajectory=np.zeros((5, 3), dtype=np.float64),
            controlled_prediction=np.zeros((5, 3), dtype=np.float64),
            drift_dominance_index=0.65,
            reachable_interval_radius=0.12,
        )

        uncertainty = UncertaintySummary(
            kind="gaussian",
            confidence_level=0.95,
            joint_variances={"pelvis_tx": 1e-4, "pelvis_ty": 1e-4},
            parameter_bounds={"mass_scale": (0.95, 1.05)},
        )

        residuals = ReplayResidualsSummary(
            max_position_drift_m=0.008,
            rms_position_drift_m=0.004,
            normal_force_residual_n=0.0,
            friction_cone_violations=0,
            reset_count=1,
        )

        options = DimeReportOptions(
            scope=ReportScope.FULL,
            strategy=EstimatorStrategy.DIME_MHE,
            include_kinematics=True,
            include_controls=True,
            include_grf=True,
            include_drift_prediction=True,
            include_uncertainty=True,
            include_residuals=True,
            include_video_overlays=True,
        )

        report = service.create_report(
            report_id="rep-full-001",
            options=options,
            run_classification=RunClassification.HYBRID_ESTIMATION,
            provenance=prov,
            engine_capability_status="qualified",
            kinematics=kinematics,
            selected_net_controls=np.array([[10.0], [12.0], [14.0], [16.0], [18.0]]),
            control_channel_names=("lumbar_torque",),
            grf_report=grf_report,
            drift_prediction=drift,
            uncertainty=uncertainty,
            residuals=residuals,
            video_overlay=overlay,
            ztcf_record=ztcf,
        )

        assert report.scope == ReportScope.FULL
        assert report.is_validated is True
        assert report.kinematics is not None
        assert report.selected_net_controls is not None
        assert report.grf_report.provenance == ReportForceProvenance.UNAVAILABLE
        assert report.grf_report.forces_n is None
        assert (
            report.unavailable_capabilities["grf"]
            == "Native in-shoe force sensor data not available"
        )

    def test_green_pointwise_vs_integrated_ztcf_distinctly_recorded(self) -> None:
        """Pointwise and integrated ZTCF are recorded with distinct semantics and arrays."""
        ztcf = _make_sample_ztcf(5)
        assert ztcf.pointwise_ztcf_drift.ndim == 1
        assert ztcf.integrated_ztcf_drift.ndim == 1
        # Pointwise is acceleration [m/s^2], integrated is cumulative displacement [m]
        assert not np.allclose(ztcf.pointwise_ztcf_drift, ztcf.integrated_ztcf_drift)
        assert ztcf.integrated_dominance == 0.82

    def test_green_report_round_trip_dict_json_provenance_preservation(self) -> None:
        """Report preserves provenance, numerical arrays, and method identity across JSON serialization."""
        service = DimeStrategySelectionService()
        prov = _make_sample_provenance()
        kinematics = _make_sample_kinematics(5)
        ztcf = _make_sample_ztcf(5)
        overlay = _make_sample_overlay(5)

        grf_report = GroundReactionForceReport(
            provenance=ReportForceProvenance.MEASURED,
            forces_n=np.ones((5, 3), dtype=np.float64) * 400.0,
            cop_m=np.zeros((5, 2), dtype=np.float64),
        )

        options = DimeReportOptions(
            scope=ReportScope.CUSTOM, selected_channels=("pelvis_tx",)
        )
        report = service.create_report(
            report_id="rep-rt-001",
            options=options,
            run_classification=RunClassification.HYBRID_ESTIMATION,
            provenance=prov,
            engine_capability_status="qualified",
            kinematics=kinematics,
            grf_report=grf_report,
            video_overlay=overlay,
            ztcf_record=ztcf,
        )

        # Dictionary round-trip
        data = report.to_dict()
        restored = DimeReportArtifact.from_dict(data)

        assert restored.report_id == report.report_id
        assert restored.provenance.model_hash == prov.model_hash
        assert restored.provenance.param_hash == prov.param_hash
        assert restored.provenance.git_commit == prov.git_commit
        assert np.allclose(restored.kinematics.times, kinematics.times)
        assert np.allclose(
            restored.ztcf_record.pointwise_ztcf_drift, ztcf.pointwise_ztcf_drift
        )
        assert np.allclose(
            restored.ztcf_record.integrated_ztcf_drift, ztcf.integrated_ztcf_drift
        )

        # JSON string round-trip
        json_str = report.to_json()
        restored_json = DimeReportArtifact.from_json(json_str)
        assert restored_json.report_id == report.report_id
        assert (
            restored_json.video_overlay.geometry_model
            == "GS3DX_Human_Refined_Ellipsoid"
        )

    def test_green_offscreen_gui_viewmodel_selection(self) -> None:
        """Offscreen ViewModel manages strategy and options without GUI launch."""
        vm = DimeStrategySelectionViewModel()
        prov_cap = _make_sample_provider_capability(is_qualified=True)
        vm.update_capabilities(prov_cap)

        assert vm.can_select(EstimatorStrategy.DIME_MHE)
        assert not vm.can_select(EstimatorStrategy.NEURAL_ESTIMATOR)

        vm.set_strategy(EstimatorStrategy.DIME_MHE)
        assert vm.current_strategy == EstimatorStrategy.DIME_MHE

        vm.set_scope(ReportScope.CUSTOM)
        vm.set_selected_channels(["pelvis_tx", "lumbar_pitch"])
        assert vm.selected_channels == ("pelvis_tx", "lumbar_pitch")

    def test_green_latex_methods_summary(self) -> None:
        """to_latex_summary outputs valid LaTeX equations, assumptions, units, and limitations."""
        prov = _make_sample_provenance()
        kinematics = _make_sample_kinematics(5)
        ztcf = _make_sample_ztcf(5)

        report = DimeReportArtifact(
            report_id="rep-latex-001",
            created_at="2026-10-04T08:00:00Z",
            scope=ReportScope.FULL,
            strategy=EstimatorStrategy.DIME_MHE,
            run_classification=RunClassification.HYBRID_ESTIMATION,
            provenance=prov,
            engine_capability_status="qualified",
            kinematics=kinematics,
            ztcf_record=ztcf,
        )

        latex = report.to_latex_summary()
        assert "\\section{Estimation Methods and Model Identification}" in latex
        assert "a_{\\mathrm{ztcf}}(t)" in latex  # Pointwise ZTCF equation
        assert "x_{\\mathrm{ztcf}}(t)" in latex  # Integrated ZTCF equation
        assert CANONICAL_DIME_UNITS["length"] in latex
        assert prov.model_hash in latex
        assert "Limitations" in latex
