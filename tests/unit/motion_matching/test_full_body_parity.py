"""TDD unit tests for cross-engine full-body parity and visual review (FB-6, #10070).

Validates:
1. CrossEngineReplayConfig contract and validation rules (DbC).
2. Numerical step-size convergence evaluator under stiff ground contact.
3. Cross-engine comparison report assembler and metric differentials.
4. Visual review frame generation for marker overlay animations.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from src.shared.python.motion_matching.cross_engine_replay import (
    CrossEngineComparisonReport,
    CrossEngineReplayConfig,
    EngineReplayOutcome,
    StepSizeConvergenceResult,
    compare_engine_replays,
    compute_step_size_convergence,
    generate_overlay_frame,
)
from src.shared.python.motion_matching.tour_metrics import SharedMetrics
from src.shared.python.motion_matching.full_body_forward_dynamics import (
    ContactAuditResult,
)

pytestmark = [pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parents[3]
CANDIDATE_PATH = (
    REPO_ROOT
    / "docs/development/simscape_tour_matching/native_evidence/two_window_fit_9967_81/returned-candidate.json"
)


def test_cross_engine_replay_config_validation() -> None:
    """Config validation rejects invalid parameters per DbC."""
    with pytest.raises(ValueError, match="duration_s must be strictly positive"):
        CrossEngineReplayConfig(
            candidate_path=CANDIDATE_PATH,
            duration_s=0.0,
            engines=("mujoco",),
        )

    with pytest.raises(ValueError, match="At least one engine must be specified"):
        CrossEngineReplayConfig(
            candidate_path=CANDIDATE_PATH,
            duration_s=1.8,
            engines=(),
        )

    with pytest.raises(FileNotFoundError, match="Candidate file not found"):
        CrossEngineReplayConfig(
            candidate_path=Path("nonexistent/candidate.json"),
            duration_s=1.8,
            engines=("mujoco",),
        )


def test_step_size_convergence_computation() -> None:
    """Step-size convergence measures max deviation and checks convergence threshold."""
    time_s = np.linspace(0.0, 1.0, 50)
    q_nom = np.zeros((50, 41))
    qd_nom = np.zeros((50, 41))

    # Add a small perturbation of 0.001 rad to refined trajectory
    q_ref = q_nom + 0.001
    qd_ref = qd_nom + 0.002

    res = compute_step_size_convergence(
        time_s=time_s,
        q_nominal=q_nom,
        qd_nominal=qd_nom,
        q_refined=q_ref,
        qd_refined=qd_ref,
        h_nominal=0.001,
        h_refined=0.0005,
        tolerance_rad=0.005,
    )

    assert isinstance(res, StepSizeConvergenceResult)
    assert res.h_nominal == 0.001
    assert res.h_refined == 0.0005
    assert pytest.approx(res.max_q_difference, abs=1e-6) == 0.001
    assert pytest.approx(res.max_qd_difference, abs=1e-6) == 0.002
    assert res.is_converged is True

    # Check non-convergence when tolerance is smaller than difference
    res_strict = compute_step_size_convergence(
        time_s=time_s,
        q_nominal=q_nom,
        qd_nominal=qd_nom,
        q_refined=q_ref,
        qd_refined=qd_ref,
        h_nominal=0.001,
        h_refined=0.0005,
        tolerance_rad=0.0005,
    )
    assert res_strict.is_converged is False


def test_cross_engine_comparison_report() -> None:
    """Cross-engine comparison report computes pairwise metric and marker differences."""
    metrics_a = SharedMetrics(
        whole_marker_rmse_m=2.84,
        early_marker_rmse_m=2.55,
        terminal_marker_rmse_m=2.68,
        club_marker_rmse_m=2.98,
        pelvis_yaw_rmse_rad=1.91,
    )
    metrics_b = SharedMetrics(
        whole_marker_rmse_m=2.86,
        early_marker_rmse_m=2.56,
        terminal_marker_rmse_m=2.70,
        club_marker_rmse_m=3.00,
        pelvis_yaw_rmse_rad=1.92,
    )

    audit = ContactAuditResult(
        max_normal_force_n=100000.0,
        max_friction_force_n=80000.0,
        max_penetration_m=0.10,
        per_sphere_max_force_n={"heel_r": 80000.0},
        per_sphere_contact_ratio={"heel_r": 0.35},
    )

    conv = StepSizeConvergenceResult(
        h_nominal=0.001,
        h_refined=0.0005,
        max_q_difference=0.002,
        max_qd_difference=0.005,
        is_converged=True,
    )

    outcome_a = EngineReplayOutcome(
        engine="mujoco",
        status="success",
        shared_metrics=metrics_a,
        contact_audit=audit,
        convergence=conv,
        max_closure_residual_m=2.67,
    )
    outcome_b = EngineReplayOutcome(
        engine="pinocchio",
        status="success",
        shared_metrics=metrics_b,
        contact_audit=audit,
        convergence=conv,
        max_closure_residual_m=2.65,
    )

    report = compare_engine_replays([outcome_a, outcome_b])
    assert isinstance(report, CrossEngineComparisonReport)
    assert len(report.engines) == 2
    assert "mujoco_vs_pinocchio" in report.pairwise_metric_diffs
    diffs = report.pairwise_metric_diffs["mujoco_vs_pinocchio"]
    assert pytest.approx(diffs["whole_marker_rmse_m_diff"], abs=1e-4) == 0.02
    assert pytest.approx(diffs["club_marker_rmse_m_diff"], abs=1e-4) == 0.02


def test_generate_overlay_frame() -> None:
    """Overlay frame generation builds visual components for target and model markers."""
    n_markers = 10
    target_m = np.zeros((n_markers, 3))
    model_m = np.ones((n_markers, 3)) * 0.05
    valid = np.ones(n_markers, dtype=bool)

    frame = generate_overlay_frame(
        frame_idx=0,
        time_s=0.0,
        target_markers_m=target_m,
        model_markers_m=model_m,
        valid=valid,
    )

    assert frame["frame_idx"] == 0
    assert frame["target_markers"].shape == (n_markers, 3)
    assert frame["model_markers"].shape == (n_markers, 3)
    assert pytest.approx(frame["rmse_m"], rel=1e-4) == np.sqrt(3 * (0.05**2))


def test_engine_replay_outcome_absent_measurements_are_none() -> None:
    """Issue #10166: EngineReplayOutcome absent unit-separated fields must be None, not 0.0."""
    from src.shared.python.motion_matching.tour_capture_contract import TourCapture
    from src.shared.python.motion_matching.full_body_forward_dynamics import (
        ContactAuditResult,
        compute_shared_metrics,
    )
    from src.shared.python.motion_matching.cross_engine_replay import (
        EngineReplayOutcome,
        StepSizeConvergenceResult,
    )

    capture = TourCapture(
        time_s=np.array([0.0, 0.01]),
        labels=("Head",),
        points_m=np.zeros((2, 1, 3)),
        valid=np.ones((2, 1), dtype=bool),
        source_sha256="test",
    )
    metrics = compute_shared_metrics(
        capture=capture,
        predicted_points_m=np.zeros((2, 1, 3)),
        tracked_labels=["Head"],
    )
    audit = ContactAuditResult(
        max_normal_force_n=0.0,
        max_friction_force_n=0.0,
        max_penetration_m=0.0,
        per_sphere_max_force_n={},
        per_sphere_contact_ratio={},
    )
    conv = StepSizeConvergenceResult(
        h_nominal=0.001,
        h_refined=0.0005,
        max_q_difference=0.002,
        max_qd_difference=0.005,
        is_converged=True,
    )

    # Outcome without explicit translation or rotation
    outcome = EngineReplayOutcome(
        engine="mujoco",
        status="success",
        shared_metrics=metrics,
        contact_audit=audit,
        convergence=conv,
        max_closure_residual_m=1.5,
    )
    assert outcome.max_closure_translation_m is None
    assert outcome.max_closure_rotation_rad is None
    data = outcome.as_dict()
    assert data["max_closure_translation_m"] is None
    assert data["max_closure_rotation_rad"] is None
    assert data["legacy_mixed_closure_residual"] == 1.5

    # Passing negative or non-finite values must raise ValueError
    with pytest.raises(ValueError, match="non-negative"):
        EngineReplayOutcome(
            engine="mujoco",
            status="success",
            shared_metrics=metrics,
            contact_audit=audit,
            convergence=conv,
            max_closure_residual_m=1.5,
            max_closure_translation_m=-0.1,
        )

    with pytest.raises(ValueError, match="non-negative"):
        EngineReplayOutcome(
            engine="mujoco",
            status="success",
            shared_metrics=metrics,
            contact_audit=audit,
            convergence=conv,
            max_closure_residual_m=1.5,
            max_closure_rotation_rad=-0.1,
        )

    with pytest.raises(ValueError, match="non-negative"):
        EngineReplayOutcome(
            engine="mujoco",
            status="success",
            shared_metrics=metrics,
            contact_audit=audit,
            convergence=conv,
            max_closure_residual_m=-1.5,
        )
