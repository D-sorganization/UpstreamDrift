"""Unit tests for the Tour Baselines Presenter (TB-11 #10596).

Validates:
1. Headless presenter states and accessible badges for every evidence status.
2. Correct capture, model selection, clocks, and marker sets.
3. Where This Came From panel generation linking hashes, preprocessing, and disclaimers.
4. Actions: Open Baseline, Clone for Experiment, Compare Models, Inspect Evidence, Reproduce.
5. Pending / unavailable / blocked models remain visible with specific reasons and issue links.
6. Viewer data preparation ensuring observed club markers cannot be confused with simulated output.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np

from src.shared.python.tour_baselines.baseline_package import (
    BackendType,
    BaselineIdentity,
    BaselinePackage,
    DynamicFeasibilityStatus,
    FitMode,
    KinematicAccuracyStatus,
    ModelTopology,
    ProductPromotionStatus,
    ScientificQualificationStatus,
    SolverConvergenceStatus,
    StatusBundle,
    export_baseline_package,
)
from src.shared.python.tour_baselines.fit_metrics import (
    MarkerMetricSummary,
    PhysicalFitMetrics,
)
from src.shared.python.tour_baselines.discovery import (
    BaselineDiscoveryService,
    SafeModelPreset,
)
from src.tools.motion_matching.tour_baselines_presenter import (
    TourBaselinesPresenter,
    TourBaselineDetailView,
    ModelItemView,
    ModelComparisonReport,
    EvidenceInspectionReport,
    WhereThisCameFromView,
)

pytestmark = [pytest.mark.unit]


def _make_test_package(
    model_id: str = "driven_double_pendulum",
    topology: ModelTopology = ModelTopology.PLANAR_DRIVEN_PENDULUM,
    club: str = "driver",
    horizon: str = "G1",
    rmse: float = 0.015,
) -> BaselinePackage:
    ident = BaselineIdentity(
        model_id=model_id,
        topology=topology,
        backend=BackendType.SCIPY_ODE,
        provider_pin="fedcba9876543210fedcba9876543210fedcba98",
        fit_mode=FitMode.TORQUE_DRIVEN,
        capture=club,
        capture_sha256="545405ccdbae87a297d16951487b501d5d76f5a2ab253cfc6d797744184943ba",
        horizon=horizon,
        frame_convention="z_up_y_forward",
        plane_convention="transverse_sagittal_frontal",
        measurement_map_version="tour-measurement-map/1.0.0",
        fixed_geometry_hash="44" * 32,
        fixed_inertia_hash="55" * 32,
        q0_hash="11" * 32,
        v0_hash="22" * 32,
        controls_hash="33" * 32,
        runtime_hashes={
            "engine_version": "1.0.0",
            "git_commit": "fedcba9876543210fedcba9876543210fedcba98",
        },
    )
    bundle = StatusBundle(
        solver_convergence=SolverConvergenceStatus.CONVERGED,
        kinematic_accuracy=KinematicAccuracyStatus.WITHIN_TOLERANCE,
        dynamic_feasibility=DynamicFeasibilityStatus.PHYSICALLY_FEASIBLE,
        scientific_qualification=ScientificQualificationStatus.QUALIFIED,
        product_promotion=ProductPromotionStatus.PROMOTED,
        has_native_replay=True,
    )
    metrics = PhysicalFitMetrics(
        whole_marker_rmse_m=rmse,
        p95_marker_error_m=rmse * 1.5,
        max_marker_error_m=rmse * 2.0,
        per_marker={
            "ClubHead": MarkerMetricSummary(
                rmse_m=rmse,
                max_m=rmse * 2.0,
                p95_m=rmse * 1.5,
                valid_count=26,
                total_count=26,
            )
        },
        per_phase={},
        endpoint_error_m=0.005,
        impact_error_m=0.008,
        in_plane_rmse_m=0.004,
        out_of_plane_residual_m=0.002,
        pelvis_yaw_rmse_rad=None,
        optimizer_weighted_loss=1.2,
        n_valid=26,
        n_excluded=0,
        total_observations=26,
        coverage_fraction=1.0,
        landmark_set_signature="sig_clubhead_26",
    )
    time_arr = np.linspace(0.0, 0.25, 26)
    q = np.zeros((26, 2))
    v = np.zeros((26, 2))
    trajs = {
        "time": time_arr,
        "q": q,
        "v": v,
        "tau": np.zeros((26, 2)),
    }
    return BaselinePackage(
        identity=ident,
        statuses=bundle,
        metrics=metrics,
        replay_command=f"python -m src.shared.python.tour_baselines.campaign --model-id {model_id} --capture {club}",
        trajectories=trajs,
        reports={
            "parameters": {"l1": 0.65, "l2": 1.05},
            "provenance": {"capture": club, "method": "trf"},
        },
        artifacts={},
        is_synthetic=False,
    )


@pytest.fixture
def baseline_store(tmp_path: Path) -> Path:
    store = tmp_path / "baselines"
    store.mkdir()
    # Export driver and iron packages
    pkg_driver = _make_test_package(
        model_id="driven_double_pendulum",
        club="driver",
        rmse=0.012,
    )
    pkg_iron = _make_test_package(
        model_id="driven_double_pendulum",
        club="iron",
        rmse=0.009,
    )
    pkg_triple = _make_test_package(
        model_id="driven_triple_pendulum",
        club="driver",
        rmse=0.015,
    )
    export_baseline_package(pkg_driver, store / "driver_pendulum.npz")
    export_baseline_package(pkg_iron, store / "iron_pendulum.npz")
    export_baseline_package(pkg_triple, store / "driver_triple.npz")
    return store


def test_presenter_lists_models_for_both_captures(baseline_store: Path) -> None:
    presenter = TourBaselinesPresenter(search_roots=[baseline_store])

    # Driver models
    driver_models = presenter.list_models(capture="driver")
    assert len(driver_models) >= 10
    model_ids = [m.model_id for m in driver_models]
    assert "driven_double_pendulum" in model_ids
    assert "full_body_mujoco" in model_ids
    assert "full_body_pinocchio" in model_ids
    assert "full_body_drake" in model_ids

    # Iron models
    iron_models = presenter.list_models(capture="iron")
    assert len(iron_models) >= 10
    assert all(m.capture in ("iron", "iron7") for m in iron_models)

    # Exact-capture package matching on roster (#10829)
    driver_triple = next(
        m for m in driver_models if m.model_id == "driven_triple_pendulum"
    )
    assert driver_triple.has_package is True

    iron_triple = next(m for m in iron_models if m.model_id == "driven_triple_pendulum")
    assert iron_triple.has_package is False


def test_presenter_badge_presentation_without_color_alone(baseline_store: Path) -> None:
    """Requirement: readable statuses without color alone; accessible text and symbols."""
    presenter = TourBaselinesPresenter(search_roots=[baseline_store])

    # 1. G1 passed model
    detail_passed = presenter.get_model_detail("full_body_mujoco", "driver")
    assert "[PASS]" in detail_passed.badge_symbol
    assert "G1 Kinematic Passed" in detail_passed.badge_text

    # 2. Rejected model (TB-04 driven_double_pendulum)
    detail_rejected = presenter.get_model_detail("driven_double_pendulum", "driver")
    assert "[REJECTED]" in detail_rejected.badge_symbol
    assert "Rejected" in detail_rejected.badge_text
    assert detail_rejected.blocker_reason is not None
    assert "DISQUALIFIED" in detail_rejected.blocker_reason

    # 3. Blocked model (myosuite)
    detail_blocked = presenter.get_model_detail("full_body_myosuite", "driver")
    assert "[BLOCKED]" in detail_blocked.badge_symbol
    assert "Blocked" in detail_blocked.badge_text
    assert detail_blocked.blocker_reason is not None

    # 4. Historical reference model
    detail_ref = presenter.get_model_detail("reconstruction_golfer", "driver")
    assert "[REF]" in detail_ref.badge_symbol
    assert "Historical Reference" in detail_ref.badge_text


def test_where_this_came_from_panel_links_hashes_and_limitations(
    baseline_store: Path,
) -> None:
    """Requirement: plain-language Where This Came From panel linking raw capture hash, preprocessing, geometry, fit config, limitations."""
    presenter = TourBaselinesPresenter(search_roots=[baseline_store])

    # Driver details
    detail_driver = presenter.get_model_detail("driven_double_pendulum", "driver")
    panel_d = detail_driver.where_this_came_from
    assert (
        panel_d.raw_capture_hash
        == "545405ccdbae87a297d16951487b501d5d76f5a2ab253cfc6d797744184943ba"
    )
    assert panel_d.capture_frequency_hz == 360.0
    assert "c3d" in panel_d.preprocessing.lower()
    assert len(panel_d.scientific_limitations) > 0
    # Must include force identifiability and holdout disclaimers
    assert any(
        "force identifiability" in lim.lower() for lim in panel_d.scientific_limitations
    )
    assert any("holdout" in lim.lower() for lim in panel_d.scientific_limitations)

    # 7-Iron details
    detail_iron = presenter.get_model_detail("driven_double_pendulum", "iron")
    panel_i = detail_iron.where_this_came_from
    assert (
        panel_i.raw_capture_hash
        == "395deb1f91006586819020fc85180409e716f07e1c680f9fb2ca114759f80845"
    )
    assert panel_i.capture_frequency_hz == 359.0


def test_viewer_data_preparation_distinguishes_observed_and_simulated(
    baseline_store: Path,
) -> None:
    """Requirement: Keep 3D and projected views distinct; observed club graphics must not be mistaken for simulated club output."""
    presenter = TourBaselinesPresenter(search_roots=[baseline_store])

    open_res = presenter.open_baseline("driven_double_pendulum", "driver")
    assert open_res.is_loaded is True
    assert open_res.view_type == "projected_2d"
    # Visual semantics distinction
    assert open_res.observed_club_visual != open_res.simulated_club_visual
    assert "marker" in open_res.observed_club_visual.lower()
    assert "continuous" in open_res.simulated_club_visual.lower()


def test_clone_for_experiment_preserves_session_directory(
    baseline_store: Path, tmp_path: Path
) -> None:
    """Requirement: Clone for Experiment creates isolated copies in user session directories without mutating parent."""
    presenter = TourBaselinesPresenter(search_roots=[baseline_store])
    session_dir = tmp_path / "user_session"

    cloned_path = presenter.clone_for_experiment(
        model_id="driven_double_pendulum",
        capture="driver",
        session_dir=session_dir,
        experiment_name="downswing_test_01",
    )
    assert cloned_path.exists()
    assert cloned_path.is_file()
    assert "user_session" in str(cloned_path)
    # Target file in baseline_store is unchanged
    assert (baseline_store / "driver_pendulum.npz").exists()


def test_compare_models_generates_metric_and_topology_diffs(
    baseline_store: Path,
) -> None:
    """Requirement: Compare Models between two baseline models / topologies."""
    presenter = TourBaselinesPresenter(search_roots=[baseline_store])

    report = presenter.compare_models(
        model_a_id="driven_double_pendulum",
        model_b_id="driven_triple_pendulum",
        capture="driver",
    )
    assert isinstance(report, ModelComparisonReport)
    assert report.model_a_id == "driven_double_pendulum"
    assert report.model_b_id == "driven_triple_pendulum"
    assert report.capture == "driver"
    assert "marker_rmse_delta_mm" in report.metric_deltas
    assert report.metric_deltas["marker_rmse_delta_mm"] == pytest.approx(3.0)
    assert "delta RMSE = +3.00 mm" in report.verdict
    assert report.topology_comparison is not None


def test_compare_models_preserves_missing_rmse_without_fabricating_delta(
    baseline_store: Path,
) -> None:
    """Requirement: Missing RMSE reports comparison unavailable instead of assuming 0.0 mm (#10810)."""
    presenter = TourBaselinesPresenter(search_roots=[baseline_store])

    # full_body_pinocchio has no package in baseline_store
    report = presenter.compare_models(
        model_a_id="driven_double_pendulum",
        model_b_id="full_body_pinocchio",
        capture="driver",
    )
    assert isinstance(report, ModelComparisonReport)
    assert "marker_rmse_delta_mm" not in report.metric_deltas
    assert report.metric_deltas == {}
    assert "comparison unavailable" in report.verdict
    assert "full_body_pinocchio lacks measured baseline package" in report.verdict

    # Both models missing package
    report_both = presenter.compare_models(
        model_a_id="full_body_drake",
        model_b_id="full_body_pinocchio",
        capture="driver",
    )
    assert "marker_rmse_delta_mm" not in report_both.metric_deltas
    assert report_both.metric_deltas == {}
    assert "comparison unavailable" in report_both.verdict
    assert "both full_body_drake and full_body_pinocchio" in report_both.verdict

    # Cross-capture fallback avoidance (#10826):
    # driven_triple_pendulum only has a Driver package in baseline_store, not Iron.
    # Comparing on Iron must not use the Driver RMSE to fabricate a cross-capture delta.
    report_cross = presenter.compare_models(
        model_a_id="driven_double_pendulum",
        model_b_id="driven_triple_pendulum",
        capture="iron",
    )
    assert "marker_rmse_delta_mm" not in report_cross.metric_deltas
    assert report_cross.metric_deltas == {}
    assert "comparison unavailable" in report_cross.verdict
    assert (
        "driven_triple_pendulum lacks measured baseline package" in report_cross.verdict
    )


def test_inspect_evidence_returns_audit_and_receipt(baseline_store: Path) -> None:
    """Requirement: Inspect Evidence shows raw receipt, status bundle, and validation logs."""
    presenter = TourBaselinesPresenter(search_roots=[baseline_store])

    evidence = presenter.inspect_evidence("driven_double_pendulum", "driver")
    assert isinstance(evidence, EvidenceInspectionReport)
    assert evidence.model_id == "driven_double_pendulum"
    assert evidence.status_bundle is not None
    assert evidence.status_bundle["convergence"] == "converged"
    assert evidence.status_bundle["feasibility"] == "physically_feasible"


def test_reproduce_returns_exact_command(baseline_store: Path) -> None:
    """Requirement: Reproduce action provides exact reproduction command."""
    presenter = TourBaselinesPresenter(search_roots=[baseline_store])

    cmd = presenter.reproduce("driven_double_pendulum", "driver")
    assert isinstance(cmd, str)
    assert len(cmd) > 0
    assert "driven_double_pendulum" in cmd
    assert "driver" in cmd
