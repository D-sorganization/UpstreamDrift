"""Unit tests for motion_pipeline.ik.geometric_backend.

Value-asserting TDD coverage for issue #7046: the geometric backend is the
single real, dependency-free IK solver. The headline test plants a known
joint angle, forward-generates marker positions from it, then asserts the
solver recovers that angle within tolerance.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.motion_pipeline.contracts import (
    JointDef,
    JointLimit,
    JointTrajectory,
    Marker,
    MarkerFrame,
    MarkerTrajectory,
    SkeletonRig,
)
from src.shared.python.motion_pipeline.ik.base import IKConfig, make_ik_solver
from src.shared.python.motion_pipeline.ik.geometric_backend import (
    GeometricIKSolver,
    forward_kinematics,
)

_TOL_RAD = math.radians(2.0)


def _make_planar_arm() -> SkeletonRig:
    """Two-link planar arm rotating about Z, lying in the XY plane.

    ``shoulder`` is the root at the origin (1 DOF, Z). ``elbow`` is offset
    1 m along +X (1 DOF, Z). ``wrist`` is a further 1 m along +X. Placing
    markers on ``elbow`` and ``wrist`` makes both the shoulder and elbow
    angles observable, so a planted pose can be recovered uniquely.
    """
    joints = {
        "shoulder": JointDef(
            name="shoulder",
            parent=None,
            children=["elbow"],
            tpose_offset=[0.0, 0.0, 0.0],
            axes=["Z"],
            limits=[JointLimit(lower=-3.14, upper=3.14)],
        ),
        "elbow": JointDef(
            name="elbow",
            parent="shoulder",
            children=["wrist"],
            tpose_offset=[1.0, 0.0, 0.0],
            axes=["Z"],
            limits=[JointLimit(lower=-3.14, upper=3.14)],
        ),
        "wrist": JointDef(
            name="wrist",
            parent="elbow",
            children=[],
            tpose_offset=[1.0, 0.0, 0.0],
            axes=["Z"],
            limits=[JointLimit(lower=-3.14, upper=3.14)],
        ),
    }
    return SkeletonRig(id="planar_arm", joints=joints, root_joint="shoulder")


def _markers_from_q(rig: SkeletonRig, q: list[float]) -> dict[str, tuple]:
    """Forward-generate marker positions (one per joint) from joint angles."""
    return forward_kinematics(rig, q)


def test_forward_kinematics_neutral_pose_matches_offsets() -> None:
    rig = _make_planar_arm()
    positions = forward_kinematics(rig, [0.0, 0.0, 0.0])
    # shoulder at origin, elbow 1 m along +X, wrist 2 m along +X.
    assert positions["shoulder"] == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)
    assert positions["elbow"] == pytest.approx((1.0, 0.0, 0.0), abs=1e-9)
    assert positions["wrist"] == pytest.approx((2.0, 0.0, 0.0), abs=1e-9)


def test_forward_kinematics_rotation_moves_child() -> None:
    rig = _make_planar_arm()
    # Rotate shoulder 90 deg about Z: elbow swings to +Y, wrist to (0, 2).
    positions = forward_kinematics(rig, [math.pi / 2.0, 0.0, 0.0])
    assert positions["elbow"] == pytest.approx((0.0, 1.0, 0.0), abs=1e-6)
    assert positions["wrist"] == pytest.approx((0.0, 2.0, 0.0), abs=1e-6)


def test_geometric_ik_recovers_planted_angles() -> None:
    """Plant shoulder/elbow angles, forward-generate markers, recover them.

    Markers are placed on the ``elbow`` and ``wrist`` joints, which makes
    the shoulder and elbow DOFs observable. The wrist's own rotation has no
    child marker, so it is unobservable and not asserted.
    """
    rig = _make_planar_arm()
    planted = [0.6, -0.4, 0.0]
    full = _markers_from_q(rig, planted)
    target_markers = {"elbow": full["elbow"], "wrist": full["wrist"]}

    solver = GeometricIKSolver(IKConfig(max_iterations=400, tolerance=1e-10))
    q = solver.solve_frame(target_markers, rig)

    assert len(q) == rig.num_dofs
    # End-effector (wrist) recovered to high precision.
    recovered = forward_kinematics(rig, q)
    assert recovered["wrist"] == pytest.approx(target_markers["wrist"], abs=1e-3)
    assert recovered["elbow"] == pytest.approx(target_markers["elbow"], abs=1e-3)
    # The two observable planted angles are recovered within tolerance.
    assert q[0] == pytest.approx(planted[0], abs=_TOL_RAD)
    assert q[1] == pytest.approx(planted[1], abs=_TOL_RAD)


def test_geometric_ik_respects_joint_limits() -> None:
    rig = _make_planar_arm()
    # Tighten the elbow limit so the solver must clamp.
    rig.joints["elbow"].limits = [JointLimit(lower=-0.1, upper=0.1)]
    full = forward_kinematics(rig, [1.0, 0.5, 0.0])
    target = {"elbow": full["elbow"], "wrist": full["wrist"]}
    solver = GeometricIKSolver()
    q = solver.solve_frame(target, rig)
    assert -0.1 - 1e-6 <= q[1] <= 0.1 + 1e-6


def test_geometric_ik_full_trajectory_returns_joint_trajectory() -> None:
    rig = _make_planar_arm()
    frames = []
    for i in range(5):
        t = i / 100.0
        q = [0.3 * math.sin(t), 0.2 * math.cos(t), 0.0]
        pos = forward_kinematics(rig, q)
        markers = {
            name: Marker(name=name, x=p[0], y=p[1], z=p[2]) for name, p in pos.items()
        }
        frames.append(MarkerFrame(timestamp=t, markers=markers, frame_index=i))
    traj = MarkerTrajectory(id="planar_traj", frames=frames)

    out = GeometricIKSolver().solve(traj, rig)
    assert isinstance(out, JointTrajectory)
    assert out.num_frames == traj.num_frames
    assert out.metadata.get("backend") == "geometric"
    for frame in out.frames:
        assert all(np.isfinite(v) for v in frame.q)


def test_make_ik_solver_geometric_returns_real_solver() -> None:
    solver = make_ik_solver("geometric")
    assert isinstance(solver, GeometricIKSolver)


def test_geometric_ik_missing_markers_raises() -> None:
    """DbC: solving with no usable marker targets raises ValueError."""
    rig = _make_planar_arm()
    solver = GeometricIKSolver()
    with pytest.raises(ValueError, match="marker"):
        solver.solve_frame({"unknown_marker": (0.0, 0.0, 0.0)}, rig)
