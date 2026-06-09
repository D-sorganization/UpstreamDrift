"""Unit tests for motion_pipeline.ik.pinocchio_backend.

After issue #7046 the Pinocchio IK backend is an unimplemented stub that
must raise ``NotImplementedError`` rather than silently returning zeros.
"""

from __future__ import annotations

import pytest

from src.shared.python.motion_pipeline.ik.pinocchio_backend import PinocchioIKSolver

from ._local_fixtures import make_3dof_phantom_rig, make_phantom_marker_trajectory


def test_pinocchio_solver_constructs_without_pinocchio() -> None:
    solver = PinocchioIKSolver()
    assert solver is not None


def test_pinocchio_solver_solve_frame_raises_not_implemented() -> None:
    rig = make_3dof_phantom_rig()
    with pytest.raises(NotImplementedError):
        PinocchioIKSolver().solve_frame({}, rig)


def test_pinocchio_solver_solve_raises_not_implemented() -> None:
    rig = make_3dof_phantom_rig()
    traj = make_phantom_marker_trajectory(num_frames=4)
    with pytest.raises(NotImplementedError):
        PinocchioIKSolver().solve(traj, rig)
