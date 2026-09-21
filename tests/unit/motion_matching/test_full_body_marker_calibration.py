"""Unit tests for full-body marker calibration and engine IK adapters (FB-4)."""

import json
from pathlib import Path
import numpy as np
import pytest

from src.shared.python.motion_matching.full_body_ik import (
    compute_marker_rms_trajectory,
    solve_full_body_ik_trajectory,
)
from src.shared.python.motion_matching.marker_calibration import (
    CalibrationResult,
    calibrate_marker_offsets,
)
from src.shared.python.motion_matching.tour_capture_contract import TourCapture

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
UPPER_PATH = (
    ROOT
    / "docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json"
)
FULL_BODY_PATH = ROOT / "docs/development/full_body_models/full_body_spec_v1.json"


@pytest.fixture
def upper_spec() -> dict:
    return json.loads(UPPER_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def fb_spec(upper_spec: dict) -> dict:
    from src.shared.python.motion_matching.full_body_spec import load_full_body_spec

    return load_full_body_spec(FULL_BODY_PATH, upper_spec)


def _require_real_pinocchio() -> None:
    try:
        import pinocchio as pin
    except ImportError as exc:
        pytest.skip(f"pinocchio not importable: {exc}")
    if (
        type(pin).__module__ == "unittest.mock"
        or not hasattr(pin, "Model")
        or not hasattr(pin, "SE3")
    ):
        pytest.skip("real pinocchio runtime required (found mock/stub)")


def _require_real_drake() -> None:
    try:
        import pydrake.all as drake_all
    except ImportError as exc:
        pytest.skip(f"pydrake not importable: {exc}")
    if type(drake_all).__module__ == "unittest.mock" or not hasattr(
        drake_all, "MultibodyPlant"
    ):
        pytest.skip("real pydrake runtime required (found mock/stub)")


def test_full_body_ik_trajectory_dbc_validation() -> None:
    labels = ("M1", "M2")
    capture = TourCapture(
        np.array([0.0, 0.01]),
        labels,
        np.zeros((2, 2, 3)),
        np.ones((2, 2), bool),
    )
    offsets = {"M1": ("B", (0.0, 0.0, 0.0)), "M2": ("B", (0.1, 0.0, 0.0))}

    def pose_fn(q: np.ndarray) -> dict:
        return {"B": (np.eye(3), q[:3])}

    # Non-1D initial_q
    with pytest.raises(ValueError, match="finite 1D array"):
        solve_full_body_ik_trajectory(pose_fn, offsets, capture, np.zeros((2, 2)))

    # Nonfinite initial_q
    with pytest.raises(ValueError, match="finite 1D array"):
        solve_full_body_ik_trajectory(
            pose_fn, offsets, capture, np.array([np.nan, 0.0])
        )

    # Missing label in offsets
    with pytest.raises(ValueError, match="missing from offsets"):
        solve_full_body_ik_trajectory(
            pose_fn,
            {"M1": ("B", (0.0, 0.0, 0.0))},
            capture,
            np.zeros(3),
        )

    # Negative closure weight
    with pytest.raises(ValueError, match="non-negative"):
        solve_full_body_ik_trajectory(
            pose_fn,
            offsets,
            capture,
            np.zeros(3),
            closure_weight=-1.0,
        )


def test_full_body_ik_trajectory_recovers_translation() -> None:
    frames = 4
    labels = ("T1", "T2", "T3")
    true_offsets = {
        "T1": ("Body1", (0.1, 0.0, 0.0)),
        "T2": ("Body1", (0.0, 0.1, 0.0)),
        "T3": ("Body1", (0.0, 0.0, 0.1)),
    }
    q_truth = np.column_stack(
        [
            np.linspace(0.1, 0.4, frames),
            np.linspace(0.2, 0.5, frames),
            np.linspace(0.3, 0.6, frames),
        ]
    )

    def pose_fn(q: np.ndarray) -> dict:
        return {"Body1": (np.eye(3), q[:3])}

    points = np.zeros((frames, 3, 3))
    for f in range(frames):
        p = pose_fn(q_truth[f])["Body1"]
        for i, lbl in enumerate(labels):
            points[f, i] = p[0] @ np.asarray(true_offsets[lbl][1]) + p[1]

    capture = TourCapture(
        np.arange(frames) / 100.0,
        labels,
        points,
        np.ones((frames, 3), bool),
    )

    q_sol = solve_full_body_ik_trajectory(
        pose_fn,
        true_offsets,
        capture,
        initial_q=np.zeros(3),
        max_nfev=25,
    )
    assert q_sol.shape == (frames, 3)
    np.testing.assert_allclose(q_sol, q_truth, atol=1e-6)

    # Verify RMS calculation
    rms_frames, per_marker_rms, total_rms = compute_marker_rms_trajectory(
        pose_fn, true_offsets, capture, q_sol
    )
    assert rms_frames.shape == (frames,)
    assert total_rms < 1e-6
    for lbl in labels:
        assert per_marker_rms[lbl] < 1e-6


def test_mujoco_full_body_ik_adapter(fb_spec: dict) -> None:
    pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.full_body_ik import MujocoFullBodyIK

    adapter = MujocoFullBodyIK(fb_spec)
    assert len(adapter.coordinate_order) == 41

    q0 = np.zeros(41)
    poses = adapter.pose_fn(q0)
    assert "Hub" in poses
    assert "Hip" in poses
    assert "femur_l" in poses
    assert "calcn_r" in poses

    for r, t in poses.values():
        assert r.shape == (3, 3)
        assert t.shape == (3,)
        np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-6)

    closure_res = adapter.closure_residuals(q0)
    assert closure_res.shape == (3,)
    assert np.isfinite(closure_res).all()

    # Test small 3-frame capture solve
    labels = ("HeadTop", "WaistLeft", "LToeIn")
    offsets = {
        "HeadTop": ("Hub", (0.0, 0.1, 0.0)),
        "WaistLeft": ("Hip", (0.1, 0.0, 0.0)),
        "LToeIn": ("calcn_l", (0.0, 0.0, 0.05)),
    }
    points = np.zeros((3, 3, 3))
    for f in range(3):
        p = adapter.pose_fn(q0)
        for i, lbl in enumerate(labels):
            body, off = offsets[lbl]
            points[f, i] = p[body][0] @ np.asarray(off) + p[body][1]

    capture = TourCapture(
        np.array([0.0, 0.01, 0.02]),
        labels,
        points,
        np.ones((3, 3), bool),
    )

    q_traj = adapter.ik_fn(offsets, capture, initial_q=q0, max_nfev=20)
    assert q_traj.shape == (3, 41)
    assert np.isfinite(q_traj).all()

    rms_frames, per_marker_rms, total_rms = adapter.evaluate_trajectory_rms(
        offsets, capture, q_traj
    )
    assert total_rms < 1e-2


def test_pinocchio_full_body_ik_adapter(fb_spec: dict) -> None:
    _require_real_pinocchio()
    from src.engines.physics_engines.pinocchio.python.full_body_ik import (
        PinocchioFullBodyIK,
    )

    adapter = PinocchioFullBodyIK(fb_spec)
    assert len(adapter.coordinate_order) == 41

    q0 = np.zeros(41)
    poses = adapter.pose_fn(q0)
    assert "Hub" in poses
    assert "Hip" in poses
    assert "femur_l" in poses
    assert "calcn_r" in poses

    closure_res = adapter.closure_residuals(q0)
    assert closure_res.shape == (3,)
    assert np.isfinite(closure_res).all()


def test_drake_full_body_ik_adapter(fb_spec: dict) -> None:
    _require_real_drake()
    from src.engines.physics_engines.drake.python.full_body_ik import DrakeFullBodyIK

    adapter = DrakeFullBodyIK(fb_spec)
    assert len(adapter.coordinate_order) == 41

    q0 = np.zeros(41)
    poses = adapter.pose_fn(q0)
    assert "Hub" in poses
    assert "Hip" in poses
    assert "femur_l" in poses
    assert "calcn_r" in poses

    closure_res = adapter.closure_residuals(q0)
    assert closure_res.shape == (3,)
    assert np.isfinite(closure_res).all()
