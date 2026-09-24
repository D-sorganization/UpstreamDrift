"""End-to-end acceptance tests for Tour Baselines user journey (TB-12, #10597).

Validates the full user journey:
Launch -> Tour Baselines -> Capture selection -> Model roster ->
Evidence inspection -> Open & visual distinction semantics ->
Session cloning -> Model comparison -> Exact CLI reproduction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from src.shared.python.tour_baselines import (
    BackendType,
    BaselineIdentity,
    BaselinePackage,
    DynamicFeasibilityStatus,
    FitMode,
    KinematicAccuracyStatus,
    MarkerMetricSummary,
    ModelTopology,
    PhaseMetricSummary,
    PhysicalFitMetrics,
    ProductPromotionStatus,
    ScientificQualificationStatus,
    SolverConvergenceStatus,
    StatusBundle,
    export_baseline_package,
)
from src.tools.motion_matching.gui import MotionMatchingWidget
from src.tools.motion_matching.tour_baselines_presenter import (
    ComputeBudgetView,
    EvidenceInspectionReport,
    ModelComparisonReport,
    TourBaselinesPresenter,
)

pytestmark = pytest.mark.unit

ROSTER_MODELS = [
    "driven_double_pendulum",
    "driven_triple_pendulum",
    "constrained_upper_body_golfer",
    "full_body_pinocchio",
    "full_body_simscape",
    "full_body_opensim",
    "full_body_mujoco",
    "full_body_drake",
    "full_body_myosuite",
]


def _make_test_package(
    model_id: str = "driven_double_pendulum",
    club: str = "driver",
    rmse: float = 0.015,
) -> BaselinePackage:
    ident = BaselineIdentity(
        model_id=model_id,
        topology=ModelTopology.PLANAR_DRIVEN_PENDULUM,
        backend=BackendType.SCIPY_ODE,
        provider_pin="fedcba9876543210fedcba9876543210fedcba98",
        fit_mode=FitMode.TORQUE_DRIVEN,
        capture=club,
        capture_sha256="545405ccdbae87a297d16951487b501d5d76f5a2ab253cfc6d797744184943ba",
        horizon="G1",
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
        per_phase={
            "address": PhaseMetricSummary(
                rmse_m=rmse * 0.5,
                max_m=rmse * 1.0,
                p95_m=rmse * 0.8,
                valid_count=5,
            ),
            "impact": PhaseMetricSummary(
                rmse_m=rmse * 1.1,
                max_m=rmse * 2.0,
                p95_m=rmse * 1.5,
                valid_count=5,
            ),
        },
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
    time_arr = np.linspace(0.0, 1.0, 26)
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


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def baseline_store(tmp_path: Path) -> Path:
    store = tmp_path / "baselines"
    store.mkdir()
    pkg_driver = _make_test_package("driven_double_pendulum", "driver", 0.012)
    pkg_iron = _make_test_package("driven_double_pendulum", "iron", 0.009)
    pkg_triple = _make_test_package("driven_triple_pendulum", "driver", 0.015)
    export_baseline_package(pkg_driver, store / "driver_pendulum.npz")
    export_baseline_package(pkg_iron, store / "iron_pendulum.npz")
    export_baseline_package(pkg_triple, store / "driver_triple.npz")
    return store


@pytest.fixture
def presenter(baseline_store: Path) -> TourBaselinesPresenter:
    return TourBaselinesPresenter(search_roots=[baseline_store])


@pytest.fixture
def widget(
    qapp: QApplication, presenter: TourBaselinesPresenter
) -> MotionMatchingWidget:
    w = MotionMatchingWidget(tb_presenter=presenter)
    yield w
    w.cleanup()


def test_roster_completeness_across_both_captures(
    presenter: TourBaselinesPresenter,
) -> None:
    """Verify all 10 roster models are discoverable across Driver and 7-Iron."""
    for cap in ("driver", "iron"):
        models = presenter.list_models(capture=cap)
        model_ids = {m.model_id for m in models}
        for expected in ROSTER_MODELS:
            assert expected in model_ids, f"Model {expected} missing for capture {cap}"

        # Verify accessible badges
        for m in models:
            assert any(
                m.badge_symbol.startswith(prefix)
                for prefix in (
                    "[PASS]",
                    "[REJECTED]",
                    "[BLOCKED]",
                    "[REF]",
                    "[CANDIDATE]",
                    "[UNQUALIFIED]",
                )
            ), f"Badge {m.badge_symbol} lacks accessible textual prefix"
            assert len(m.name) > 0
            assert len(m.ownership) > 0


def test_model_detail_and_provenance_integrity(
    presenter: TourBaselinesPresenter,
) -> None:
    """Verify detailed views, compute budgets, and provenance panels."""
    for cap in ("driver", "iron"):
        for model_id in ROSTER_MODELS:
            detail = presenter.get_model_detail(model_id, cap)
            assert detail.model_id == model_id
            assert detail.capture == cap
            assert len(detail.badge_text) > 0
            assert len(detail.ownership) > 0

            # Compute budget verification
            budget = presenter.get_compute_budget(model_id)
            assert isinstance(budget, ComputeBudgetView)
            assert budget.max_wall_clock_s > 0
            assert budget.max_evaluations > 0
            assert budget.parameter_dimension > 0

            # Provenance & lineage verification
            where = detail.where_this_came_from
            assert len(where.raw_capture_hash) == 64
            assert where.capture_frequency_hz in (359, 360)
            assert len(where.preprocessing) > 0
            assert len(where.geometry_spec) > 0
            assert len(where.scientific_limitations) > 0


def test_visual_distinction_semantics(presenter: TourBaselinesPresenter) -> None:
    """Verify explicit visual distinctions: 3D vs 2D, and marker points vs continuous mesh."""
    # Pendulum is projected 2D
    open_2d = presenter.open_baseline("driven_double_pendulum", "driver")
    assert open_2d.view_type == "projected_2d"
    assert "marker_points" in open_2d.observed_club_visual
    assert "continuous_mesh" in open_2d.simulated_club_visual

    # Full-body is spatial 3D
    open_3d = presenter.open_baseline("full_body_pinocchio", "driver")
    assert open_3d.view_type == "3d"
    assert "marker_points" in open_3d.observed_club_visual
    assert "continuous_mesh" in open_3d.simulated_club_visual


def test_session_cloning_isolation(
    presenter: TourBaselinesPresenter, tmp_path: Path
) -> None:
    """Verify session cloning produces isolated, valid presets in session dir."""
    session_dir = tmp_path / "cloned_session"
    cloned_file = presenter.clone_for_experiment(
        model_id="driven_double_pendulum",
        capture="driver",
        session_dir=session_dir,
        experiment_name="acceptance_test_exp",
    )
    assert cloned_file.exists()
    assert cloned_file.is_file()
    assert "cloned_session" in str(cloned_file)


def test_model_comparison_across_complexities(
    presenter: TourBaselinesPresenter,
) -> None:
    """Verify model comparison produces deltas when measured and handles unmeasured packages honestly."""
    # Both measured: produces delta and topological explanations
    report = presenter.compare_models(
        model_a_id="driven_double_pendulum",
        model_b_id="driven_triple_pendulum",
        capture="driver",
    )
    assert isinstance(report, ModelComparisonReport)
    assert len(report.verdict) > 0
    assert "marker_rmse_delta_mm" in report.metric_deltas
    assert report.metric_deltas["marker_rmse_delta_mm"] == pytest.approx(3.0)
    assert "delta RMSE = +3.00 mm" in report.verdict
    assert "model_a_ownership" in report.topology_comparison
    assert "model_b_ownership" in report.topology_comparison

    # Model lacking package: reports comparison unavailable without computing false delta (#10810)
    report_unmeasured = presenter.compare_models(
        model_a_id="driven_double_pendulum",
        model_b_id="full_body_pinocchio",
        capture="driver",
    )
    assert isinstance(report_unmeasured, ModelComparisonReport)
    assert "marker_rmse_delta_mm" not in report_unmeasured.metric_deltas
    assert report_unmeasured.metric_deltas == {}
    assert "comparison unavailable" in report_unmeasured.verdict
    assert (
        "full_body_pinocchio lacks measured baseline package"
        in report_unmeasured.verdict
    )


def test_evidence_inspection_and_audit_receipt(
    presenter: TourBaselinesPresenter,
) -> None:
    """Verify evidence inspection includes commit hashes, engine versions, and status bundle."""
    for model_id in ("driven_double_pendulum", "full_body_pinocchio"):
        evidence = presenter.inspect_evidence(model_id, "driver")
        assert isinstance(evidence, EvidenceInspectionReport)
        assert len(evidence.git_commit) > 0
        assert len(evidence.engine_version) > 0
        assert "qualification" in evidence.status_bundle
        assert "feasibility" in evidence.status_bundle


def test_cli_reproduction_commands(presenter: TourBaselinesPresenter) -> None:
    """Verify exact copyable CLI commands for clean-environment reproduction."""
    for model_id in ROSTER_MODELS:
        cmd = presenter.reproduce(model_id, "driver")
        assert "--model" in cmd
        assert model_id in cmd
        assert "--capture driver" in cmd
        assert "--reproduce" in cmd


def test_full_ui_journey(widget: MotionMatchingWidget, tmp_path: Path) -> None:
    """Test full UI interaction journey in MotionMatchingWidget."""
    # 1. Switch to Tour Baselines tab (Tab 5)
    widget.tabs.setCurrentIndex(4)
    assert widget.tabs.tabText(4) == "Tour Baselines"

    # 2. Verify initial driver models and details
    assert widget.tb_capture.currentText() == "Driver"
    assert widget.tb_model.count() >= 10
    assert "[PASS]" in widget.tb_badge.text() or "[" in widget.tb_badge.text()

    # 3. Switch to 7-Iron
    widget.tb_capture.setCurrentText("7-Iron")
    assert "359" in widget.tb_freq.text() or "360" in widget.tb_freq.text()

    # 4. Actions: Open, Compare, Evidence, Reproduce
    widget._on_tb_open()
    assert "[OPEN]" in widget.tb_log.toPlainText()

    widget._on_tb_compare()
    assert "[COMPARE]" in widget.tb_log.toPlainText()

    widget._on_tb_inspect_evidence()
    assert "[EVIDENCE]" in widget.tb_log.toPlainText()

    widget._on_tb_reproduce()
    assert "[REPRODUCE]" in widget.tb_log.toPlainText()
