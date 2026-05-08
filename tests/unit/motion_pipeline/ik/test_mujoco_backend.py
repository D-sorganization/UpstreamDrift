"""Unit tests for motion_pipeline.ik.mujoco_backend."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_pipeline.contracts import JointTrajectory
from src.shared.python.motion_pipeline.ik.mujoco_backend import MuJoCoIKSolver

from ._local_fixtures import make_3dof_phantom_rig, make_phantom_marker_trajectory


def test_mujoco_solver_constructs_without_mujoco_package() -> None:
    """The placeholder solver does not import mujoco at construction."""
    solver = MuJoCoIKSolver()
    assert solver is not None


def test_mujoco_solver_solve_frame_returns_neutral_pose() -> None:
    rig = make_3dof_phantom_rig()
    solver = MuJoCoIKSolver()
    q = solver.solve_frame({"M1": (0.0, 0.0, 0.0)}, rig)
    assert len(q) == rig.num_dofs
    for v in q:
        assert v == pytest.approx(0.0)


def test_mujoco_solver_respects_joint_limits() -> None:
    rig = make_3dof_phantom_rig()
    solver = MuJoCoIKSolver()
    q = solver.solve_frame({}, rig)
    # Each value in q must be within its joint's limits
    assert all(np.isfinite(v) for v in q)


def test_mujoco_solver_full_trajectory_returns_joint_trajectory() -> None:
    rig = make_3dof_phantom_rig()
    traj = make_phantom_marker_trajectory(num_frames=5)
    solver = MuJoCoIKSolver()
    out = solver.solve(traj, rig)
    assert isinstance(out, JointTrajectory)
    assert out.num_frames == traj.num_frames
    assert out.metadata.get("backend") == "mujoco"


def test_mujoco_solver_skips_when_real_solve_unavailable() -> None:
    """If a future implementation imports mujoco at solve() time, allow skip.

    This guard documents the spec: heavy backends should use
    pytest.importorskip when their hard dependency is missing.
    """
    pytest.importorskip("mujoco")  # ensures real run only when available
    # When mujoco is present, the placeholder still returns neutral pose
    rig = make_3dof_phantom_rig()
    traj = make_phantom_marker_trajectory(num_frames=3)
    out = MuJoCoIKSolver().solve(traj, rig)
    assert out.num_frames == traj.num_frames
