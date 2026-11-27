"""#14756 P4: cross_engine_replay.to_parity_report must not fake pointwise parity (#10350)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_matching.cross_engine_replay import (
    EngineReplayOutcome,
    StepSizeConvergenceResult,
    compare_engine_replays,
)
from src.shared.python.motion_matching.tour_metrics import SharedMetrics

pytestmark = pytest.mark.unit


def _convergence() -> StepSizeConvergenceResult:
    return StepSizeConvergenceResult(
        h_nominal=0.001,
        h_refined=0.0005,
        max_q_difference=0.0,
        max_qd_difference=0.0,
        is_converged=True,
    )


def _shared_metrics(rmse_m: float) -> SharedMetrics:
    return SharedMetrics(
        whole_marker_rmse_m=rmse_m,
        early_marker_rmse_m=rmse_m,
        terminal_marker_rmse_m=rmse_m,
        club_marker_rmse_m=rmse_m,
        pelvis_yaw_rmse_rad=0.01,
    )


def test_to_parity_report_omits_pointwise_without_trajectories() -> None:
    """Aggregate RMSE parity must not populate pointwise_differences rows."""
    same_rmse = 0.10
    outcomes = (
        EngineReplayOutcome(
            engine="mujoco",
            status="PASSED",
            shared_metrics=_shared_metrics(same_rmse),
            contact_audit=None,
            convergence=_convergence(),
            max_closure_residual_m=0.0,
        ),
        EngineReplayOutcome(
            engine="drake",
            status="PASSED",
            shared_metrics=_shared_metrics(same_rmse),
            contact_audit=None,
            convergence=_convergence(),
            max_closure_residual_m=0.0,
        ),
    )
    report = compare_engine_replays(outcomes)
    assert report.max_marker_rmse_diff_m == pytest.approx(0.0)

    parity = report.to_parity_report()
    for row in parity.engine_rows.values():
        assert row.pointwise_differences == {}


def test_to_parity_report_uses_aligned_trajectories_not_aggregate_rmse() -> None:
    """Equal aggregate RMSE with divergent aligned trajectories must fail pointwise."""
    n_frames, n_markers = 10, 4
    traj_a = np.zeros((n_frames, n_markers, 3), dtype=np.float64)
    traj_b = np.zeros((n_frames, n_markers, 3), dtype=np.float64)
    traj_a[..., 0] = 0.10
    traj_b[..., 0] = -0.10

    aggregate_rmse = float(np.sqrt(np.mean(traj_a**2)))
    assert aggregate_rmse == pytest.approx(float(np.sqrt(np.mean(traj_b**2))))

    outcomes = (
        EngineReplayOutcome(
            engine="mujoco",
            status="PASSED",
            shared_metrics=_shared_metrics(aggregate_rmse),
            contact_audit=None,
            convergence=_convergence(),
            max_closure_residual_m=0.0,
            aligned_model_markers_m=traj_a,
        ),
        EngineReplayOutcome(
            engine="drake",
            status="PASSED",
            shared_metrics=_shared_metrics(aggregate_rmse),
            contact_audit=None,
            convergence=_convergence(),
            max_closure_residual_m=0.0,
            aligned_model_markers_m=traj_b,
        ),
    )
    report = compare_engine_replays(outcomes)
    parity = report.to_parity_report()

    mujoco_pt = parity.engine_rows["mujoco"].pointwise_differences
    assert mujoco_pt
    diff = next(iter(mujoco_pt.values()))
    assert diff.pass_gate is False
    assert pytest.approx(diff.max_abs_diff) == 0.20
