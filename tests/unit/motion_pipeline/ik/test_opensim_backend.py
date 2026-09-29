"""Unit tests for motion_pipeline.ik.opensim_backend.

After issue #7046 the OpenSim IK backend is an unimplemented stub.
``solve()`` raises ImportError when opensim is missing (its hard
dependency); ``solve_frame()`` always raises ``NotImplementedError``.
"""

from __future__ import annotations

import pytest

from src.shared.python.motion_pipeline.ik.opensim_backend import OpenSimIKSolver

from ._local_fixtures import make_3dof_phantom_rig, make_phantom_marker_trajectory


def test_opensim_solver_constructs_without_opensim() -> None:
    """Construction does not require opensim to be installed."""
    solver = OpenSimIKSolver()
    assert solver is not None


def test_opensim_solver_solve_raises_without_opensim() -> None:
    """solve() imports opensim first and raises ImportError if missing.

    When opensim *is* installed, solve() delegates to the unimplemented
    solve_frame() and surfaces NotImplementedError instead.
    """
    rig = make_3dof_phantom_rig()
    traj = make_phantom_marker_trajectory(num_frames=2)
    try:
        import opensim  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="OpenSim"):
            OpenSimIKSolver().solve(traj, rig)
        return
    with pytest.raises(NotImplementedError):
        OpenSimIKSolver().solve(traj, rig)


def test_opensim_solve_frame_raises_not_implemented() -> None:
    rig = make_3dof_phantom_rig()
    with pytest.raises(NotImplementedError):
        OpenSimIKSolver().solve_frame({}, rig)
