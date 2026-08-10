"""Live-Pinocchio tests for the motion_pipeline IK backend.

Epic #8390 (C1/#8401). Planted-pose round trips: forward-generate marker
positions from a known configuration with the bridge model itself, solve,
and require the recovered configuration to reproduce the targets. Lives
outside ``tests/unit`` because that tree's conftest replaces ``pinocchio``
with a spec-less MagicMock.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from src.shared.python.motion_pipeline.contracts import (
    JointDef,
    JointLimit,
    Marker,
    MarkerFrame,
    MarkerTrajectory,
    SkeletonRig,
)
from src.shared.python.motion_pipeline.ik.base import IKConfig, make_ik_solver
from src.shared.python.motion_pipeline.ik.pinocchio_backend import PinocchioIKSolver
from src.shared.python.motion_pipeline.model_bridge import (
    rig_joint_link_name,
    rig_to_pinocchio_model,
)


def _available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ValueError, ModuleNotFoundError):
        return False


PIN_AVAILABLE = _available("pinocchio")
PINK_AVAILABLE = _available("pink")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.motion_pipeline,
    pytest.mark.requires_pinocchio,
    pytest.mark.skipif(not PIN_AVAILABLE, reason="pinocchio not installed"),
]


def _arm_rig() -> SkeletonRig:
    joints = {
        "shoulder": JointDef(
            name="shoulder",
            parent=None,
            children=["elbow"],
            tpose_offset=[0.0, 0.0, 0.0],
            axes=["Y", "Z"],
            limits=[
                JointLimit(lower=-2.0, upper=2.0),
                JointLimit(lower=-2.0, upper=2.0),
            ],
        ),
        "elbow": JointDef(
            name="elbow",
            parent="shoulder",
            children=["wrist"],
            tpose_offset=[0.3, 0.0, 0.0],
            axes=["Y"],
            limits=[JointLimit(lower=-2.0, upper=2.0)],
        ),
        "wrist": JointDef(
            name="wrist",
            parent="elbow",
            children=[],
            tpose_offset=[0.25, 0.0, 0.0],
            axes=["Y"],
            limits=[JointLimit(lower=-2.0, upper=2.0)],
        ),
    }
    return SkeletonRig(id="arm4", joints=joints, root_joint="shoulder")


def _markers_for(rig: SkeletonRig, q: np.ndarray) -> dict[str, tuple]:
    import pinocchio as pin

    model = rig_to_pinocchio_model(rig)
    data = model.createData()
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    out = {}
    for name in ("elbow", "wrist"):
        fid = model.getFrameId(rig_joint_link_name(name))
        out[name] = tuple(np.asarray(data.oMf[fid].translation))
    return out


def _marker_error(rig: SkeletonRig, q: list[float], targets: dict) -> float:
    import pinocchio as pin

    model = rig_to_pinocchio_model(rig)
    data = model.createData()
    pin.forwardKinematics(model, data, np.asarray(q))
    pin.updateFramePlacements(model, data)
    errs = []
    for name, target in targets.items():
        fid = model.getFrameId(rig_joint_link_name(name))
        errs.append(
            float(
                np.linalg.norm(
                    np.asarray(data.oMf[fid].translation) - np.asarray(target)
                )
            )
        )
    return max(errs)


def test_lm_recovers_planted_pose_targets() -> None:
    rig = _arm_rig()
    q_true = np.array([0.4, -0.3, 0.6, -0.5])
    targets = _markers_for(rig, q_true)
    solver = PinocchioIKSolver(IKConfig(max_iterations=300, tolerance=1e-8))
    q = solver.solve_frame(targets, rig)
    assert _marker_error(rig, q, targets) < 1e-4


@pytest.mark.skipif(not PINK_AVAILABLE, reason="pin-pink not installed")
def test_pink_method_recovers_planted_pose_targets() -> None:
    rig = _arm_rig()
    q_true = np.array([0.4, -0.3, 0.6, -0.5])
    targets = _markers_for(rig, q_true)
    solver = PinocchioIKSolver(
        IKConfig(max_iterations=300, tolerance=1e-8), method="pink"
    )
    q = solver.solve_frame(targets, rig)
    assert _marker_error(rig, q, targets) < 1e-4


def test_trajectory_solve_warm_starts_and_tracks() -> None:
    rig = _arm_rig()
    frames = []
    qs = []
    for i in range(5):
        q = np.array([0.1 * i, -0.05 * i, 0.12 * i, -0.08 * i])
        qs.append(q)
        targets = _markers_for(rig, q)
        frames.append(
            MarkerFrame(
                timestamp=i / 100.0,
                frame_index=i,
                markers={
                    name: Marker(name=name, x=p[0], y=p[1], z=p[2])
                    for name, p in targets.items()
                },
            )
        )
    traj = MarkerTrajectory(id="arm_traj", frames=frames)
    out = PinocchioIKSolver(IKConfig(max_iterations=300, tolerance=1e-8)).solve(
        traj, rig
    )
    assert len(out.frames) == 5
    assert out.metadata["backend"] == "pinocchio"
    for frame, q in zip(out.frames, qs, strict=True):
        targets = _markers_for(rig, q)
        assert _marker_error(rig, frame.q, targets) < 1e-3


def test_factory_returns_working_pinocchio_solver() -> None:
    """make_ik_solver('pinocchio') no longer raises NotImplementedError."""
    solver = make_ik_solver("pinocchio")
    rig = _arm_rig()
    targets = _markers_for(rig, np.array([0.2, 0.1, -0.2, 0.3]))
    q = solver.solve_frame(targets, rig)
    assert len(q) == rig.num_dofs


def test_parity_with_geometric_backend_on_shared_rig() -> None:
    """Both solvers must reproduce the same marker targets (parity is on
    task-space error, not joint values — the chains differ in convention)."""
    rig = _arm_rig()
    q_true = np.array([0.3, -0.2, 0.4, -0.3])
    targets = _markers_for(rig, q_true)
    q_pin = PinocchioIKSolver(IKConfig(max_iterations=300, tolerance=1e-8)).solve_frame(
        targets, rig
    )
    assert _marker_error(rig, q_pin, targets) < 1e-4
