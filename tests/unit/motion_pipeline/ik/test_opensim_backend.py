"""Unit tests for motion_pipeline.ik.opensim_backend."""

from __future__ import annotations

import pytest

from src.shared.python.motion_pipeline.ik.opensim_backend import OpenSimIKSolver

from ._local_fixtures import make_3dof_phantom_rig, make_phantom_marker_trajectory


def test_opensim_solver_constructs_without_opensim() -> None:
    """Construction does not require opensim to be installed."""
    solver = OpenSimIKSolver()
    assert solver is not None


def test_opensim_solver_solve_raises_or_skips_without_opensim() -> None:
    """solve() imports opensim and should raise ImportError if missing."""
    rig = make_3dof_phantom_rig()
    traj = make_phantom_marker_trajectory(num_frames=2)
    try:
        import opensim  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="OpenSim"):
            OpenSimIKSolver().solve(traj, rig)
        return
    # If opensim is present, solve() should produce a trajectory
    out = OpenSimIKSolver().solve(traj, rig)
    assert out.num_frames == traj.num_frames


def test_opensim_solve_frame_returns_neutral_pose() -> None:
    """solve_frame is a placeholder that doesn't import opensim."""
    rig = make_3dof_phantom_rig()
    q = OpenSimIKSolver().solve_frame({}, rig)
    assert len(q) == rig.num_dofs
