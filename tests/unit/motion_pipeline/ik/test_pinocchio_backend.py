"""Unit tests for motion_pipeline.ik.pinocchio_backend.

Implemented under epic #8390 (C1/#8401), closing the #7046 stub. This
tree's conftest installs a spec-less ``pinocchio`` MagicMock, so the
backend must report the dependency as missing here (ImportError with an
install hint) — its real solves are covered by
``tests/integration/motion_pipeline/test_pinocchio_ik_live.py``.
"""

from __future__ import annotations

import pytest

from src.shared.python.motion_pipeline.ik.pinocchio_backend import PinocchioIKSolver

from ._local_fixtures import make_3dof_phantom_rig, make_phantom_marker_trajectory


def test_pinocchio_solver_constructs_without_pinocchio() -> None:
    solver = PinocchioIKSolver()
    assert solver is not None
    assert solver.method == "lm"


def test_invalid_method_rejected() -> None:
    with pytest.raises(ValueError, match="method"):
        PinocchioIKSolver(method="nope")


def test_negative_damping_rejected() -> None:
    with pytest.raises(ValueError, match="damping"):
        PinocchioIKSolver(damping=-1.0)


def test_solve_frame_reports_missing_dependency() -> None:
    rig = make_3dof_phantom_rig()
    with pytest.raises(ImportError, match="pinocchio"):
        PinocchioIKSolver().solve_frame({}, rig)


def test_solve_reports_missing_dependency() -> None:
    rig = make_3dof_phantom_rig()
    traj = make_phantom_marker_trajectory(num_frames=4)
    with pytest.raises(ImportError, match="pinocchio"):
        PinocchioIKSolver().solve(traj, rig)
