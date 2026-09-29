"""Unit tests for motion_pipeline.ik.mujoco_backend.

After issue #7046 the MuJoCo IK backend is an unimplemented stub that must
raise ``NotImplementedError`` rather than silently returning a neutral pose.
"""

from __future__ import annotations

import pytest

from src.shared.python.motion_pipeline.ik.mujoco_backend import MuJoCoIKSolver

from ._local_fixtures import make_3dof_phantom_rig, make_phantom_marker_trajectory


def test_mujoco_solver_constructs_without_mujoco_package() -> None:
    """The stub solver does not import mujoco at construction."""
    solver = MuJoCoIKSolver()
    assert solver is not None


def test_mujoco_solver_solve_frame_raises_not_implemented() -> None:
    rig = make_3dof_phantom_rig()
    solver = MuJoCoIKSolver()
    with pytest.raises(NotImplementedError):
        solver.solve_frame({"M1": (0.0, 0.0, 0.0)}, rig)


def test_mujoco_solver_solve_raises_not_implemented() -> None:
    rig = make_3dof_phantom_rig()
    traj = make_phantom_marker_trajectory(num_frames=5)
    solver = MuJoCoIKSolver()
    with pytest.raises(NotImplementedError):
        solver.solve(traj, rig)
