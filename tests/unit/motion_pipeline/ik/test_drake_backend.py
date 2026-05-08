"""Unit tests for motion_pipeline.ik.drake_backend."""

from __future__ import annotations

import pytest

from src.shared.python.motion_pipeline.contracts import JointTrajectory
from src.shared.python.motion_pipeline.ik.drake_backend import DrakeIKSolver

from ._local_fixtures import make_3dof_phantom_rig, make_phantom_marker_trajectory


def test_drake_solver_constructs() -> None:
    solver = DrakeIKSolver()
    assert solver is not None


def test_drake_solver_solve_frame_returns_neutral_pose() -> None:
    rig = make_3dof_phantom_rig()
    solver = DrakeIKSolver()
    q = solver.solve_frame({}, rig)
    assert len(q) == rig.num_dofs


def test_drake_solver_full_trajectory() -> None:
    rig = make_3dof_phantom_rig()
    traj = make_phantom_marker_trajectory(num_frames=4)
    solver = DrakeIKSolver()
    out = solver.solve(traj, rig)
    assert isinstance(out, JointTrajectory)
    assert out.num_frames == traj.num_frames


def test_drake_solver_with_pydrake_present() -> None:
    """Spec calls for pytest.importorskip("pydrake") when full impl arrives."""
    pytest.importorskip("pydrake")
    rig = make_3dof_phantom_rig()
    traj = make_phantom_marker_trajectory(num_frames=3)
    out = DrakeIKSolver().solve(traj, rig)
    assert out.num_frames == traj.num_frames
