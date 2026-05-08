"""Unit tests for motion_pipeline.ik.pinocchio_backend."""

from __future__ import annotations

import pytest

from src.shared.python.motion_pipeline.contracts import JointTrajectory
from src.shared.python.motion_pipeline.ik.pinocchio_backend import PinocchioIKSolver

from ._local_fixtures import make_3dof_phantom_rig, make_phantom_marker_trajectory


def test_pinocchio_solver_constructs_without_pinocchio() -> None:
    solver = PinocchioIKSolver()
    assert solver is not None


def test_pinocchio_solver_solve_frame_returns_neutral_pose() -> None:
    rig = make_3dof_phantom_rig()
    q = PinocchioIKSolver().solve_frame({}, rig)
    assert len(q) == rig.num_dofs


def test_pinocchio_solver_full_trajectory() -> None:
    rig = make_3dof_phantom_rig()
    traj = make_phantom_marker_trajectory(num_frames=4)
    out = PinocchioIKSolver().solve(traj, rig)
    assert isinstance(out, JointTrajectory)
    assert out.num_frames == traj.num_frames


def test_pinocchio_solver_with_pinocchio_present() -> None:
    pytest.importorskip("pinocchio")
    rig = make_3dof_phantom_rig()
    traj = make_phantom_marker_trajectory(num_frames=3)
    out = PinocchioIKSolver().solve(traj, rig)
    assert out.num_frames == traj.num_frames
