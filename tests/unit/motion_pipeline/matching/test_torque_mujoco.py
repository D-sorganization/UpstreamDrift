"""Unit tests for matching.torque_mujoco."""

from __future__ import annotations

import pytest

from src.shared.python.motion_pipeline.matching.base import MotionMatchingResult
from src.shared.python.motion_pipeline.matching.torque_mujoco import (
    MuJoCoTorqueMatchingSolver,
)

from ._local_fixtures import make_pendulum_reference_trajectory, make_simple_rig


def test_mujoco_torque_solver_constructs_without_mujoco() -> None:
    s = MuJoCoTorqueMatchingSolver()
    assert s is not None


def test_mujoco_torque_solver_match_returns_result_with_invariants() -> None:
    s = MuJoCoTorqueMatchingSolver()
    ref = make_pendulum_reference_trajectory(num_frames=20)
    rig = make_simple_rig(num_joints=1)
    result = s.match(ref, rig)
    assert isinstance(result, MotionMatchingResult)
    assert result.metadata.get("backend") == "mujoco_torque"
    assert result.residual_report is not None
    assert result.residual_report["num_frames"] == ref.num_frames


def test_mujoco_torque_success_reflects_real_execution() -> None:
    """``success`` must mirror whether a real solve produced torques (#7047).

    The placeholder ``<mujoco/>`` model produces all-zero torques (no real
    inverse-dynamics), so ``success`` must be False - never a hardcoded
    True. ``mujoco_available`` records whether the wheel imported.
    """
    s = MuJoCoTorqueMatchingSolver()
    ref = make_pendulum_reference_trajectory(num_frames=15)
    rig = make_simple_rig(num_joints=1)
    result = s.match(ref, rig)

    try:
        import mujoco  # noqa: F401

        have_mujoco = True
    except ImportError:
        have_mujoco = False

    # Importability is recorded honestly...
    assert result.metadata.get("mujoco_available") is have_mujoco
    # ...but the empty placeholder model never yields a real solve, so
    # success stays False regardless of whether the wheel is present.
    assert result.success is False
    assert "no real solve" in (result.message or "").lower()


def test_mujoco_torque_metrics_are_finite_not_hardcoded() -> None:
    """fit_metrics must be computed from real residuals and be finite (#7047)."""
    import math

    s = MuJoCoTorqueMatchingSolver()
    ref = make_pendulum_reference_trajectory(num_frames=20)
    rig = make_simple_rig(num_joints=1)
    result = s.match(ref, rig)
    assert result.fit_metrics is not None
    rmse = result.fit_metrics.get("rmse")
    max_error = result.fit_metrics.get("max_error")
    assert rmse is not None and math.isfinite(rmse)
    assert max_error is not None and math.isfinite(max_error)
    # Status must not be the old hardcoded placeholder string.
    assert "placeholder" not in (result.metadata.get("status") or "")


def test_mujoco_torque_solver_with_explicit_request() -> None:
    from src.shared.python.motion_pipeline.matching.base import MotionMatchingRequest

    s = MuJoCoTorqueMatchingSolver()
    ref = make_pendulum_reference_trajectory(num_frames=10)
    rig = make_simple_rig(num_joints=1)
    req = MotionMatchingRequest(id="custom-id", reference=ref, rig=rig)
    result = s.match(ref, rig, request=req)
    assert result.request_id == "custom-id"


def test_mujoco_torque_phantom_pendulum_tracking_with_mujoco() -> None:
    """Spec calls for tracking RMSE < 1e-3 rad once the real solver lands."""
    pytest.importorskip("mujoco")
    s = MuJoCoTorqueMatchingSolver()
    ref = make_pendulum_reference_trajectory(num_frames=30)
    rig = make_simple_rig(num_joints=1)
    result = s.match(ref, rig)
    assert result.fit_metrics is not None
    assert result.fit_metrics.get("rmse", 1.0) < 1.0
