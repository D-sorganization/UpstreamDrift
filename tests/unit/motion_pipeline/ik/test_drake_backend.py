"""Unit tests for motion_pipeline.ik.drake_backend.

After issue #7046 the Drake IK backend is an unimplemented stub that must
raise ``NotImplementedError`` rather than silently returning a neutral pose.
"""

from __future__ import annotations

import pytest

from src.shared.python.motion_pipeline.ik.drake_backend import DrakeIKSolver

from ._local_fixtures import make_3dof_phantom_rig, make_phantom_marker_trajectory


def test_drake_solver_constructs() -> None:
    solver = DrakeIKSolver()
    assert solver is not None


def test_drake_solver_solve_frame_raises_not_implemented() -> None:
    rig = make_3dof_phantom_rig()
    solver = DrakeIKSolver()
    with pytest.raises(NotImplementedError):
        solver.solve_frame({}, rig)


def test_drake_solver_solve_raises_not_implemented() -> None:
    rig = make_3dof_phantom_rig()
    traj = make_phantom_marker_trajectory(num_frames=4)
    solver = DrakeIKSolver()
    with pytest.raises(NotImplementedError):
        solver.solve(traj, rig)
