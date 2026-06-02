"""Stub IK backends must raise NotImplementedError (issue #7046).

The mujoco/drake/opensim/pinocchio IK backends are not yet implemented.
They must fail loudly via ``NotImplementedError`` from ``solve_frame``
rather than silently returning a neutral zero pose, which previously
masked the missing implementation end-to-end.
"""

from __future__ import annotations

import pytest

from src.shared.python.motion_pipeline.ik.drake_backend import DrakeIKSolver
from src.shared.python.motion_pipeline.ik.mujoco_backend import MuJoCoIKSolver
from src.shared.python.motion_pipeline.ik.opensim_backend import OpenSimIKSolver
from src.shared.python.motion_pipeline.ik.pinocchio_backend import PinocchioIKSolver

from ._local_fixtures import make_3dof_phantom_rig, make_phantom_marker_trajectory

_STUB_SOLVERS = [
    ("mujoco", MuJoCoIKSolver),
    ("drake", DrakeIKSolver),
    ("opensim", OpenSimIKSolver),
    ("pinocchio", PinocchioIKSolver),
]


@pytest.mark.parametrize("name,cls", _STUB_SOLVERS)
def test_stub_solve_frame_raises_not_implemented(name: str, cls: type) -> None:
    rig = make_3dof_phantom_rig()
    with pytest.raises(NotImplementedError):
        cls().solve_frame({}, rig)


@pytest.mark.parametrize("name,cls", _STUB_SOLVERS)
def test_stub_solve_raises(name: str, cls: type) -> None:
    """``solve()`` must raise loudly - never return silent zeros.

    The OpenSim backend imports its hard dependency first, so when opensim
    is absent it raises ImportError before reaching the unimplemented
    ``solve_frame``; otherwise all stubs raise NotImplementedError.
    """
    rig = make_3dof_phantom_rig()
    traj = make_phantom_marker_trajectory(num_frames=2)
    with pytest.raises((NotImplementedError, ImportError)):
        cls().solve(traj, rig)
