"""Stub IK backends must raise NotImplementedError (issue #7046).

The mujoco/drake/opensim IK backends are not yet implemented. They must
fail loudly via ``NotImplementedError`` from ``solve_frame`` rather than
silently returning a neutral zero pose, which previously masked the
missing implementation end-to-end.

The pinocchio backend is implemented as of epic #8390 (C1/#8401); under
this tree's conftest (which installs a spec-less pinocchio mock) it must
raise ``ImportError`` with an install hint, mirroring the OpenSim pattern.
Its real solves are covered by
``tests/integration/motion_pipeline/test_pinocchio_ik_live.py``.
"""

from __future__ import annotations

import pytest

from src.shared.python.motion_pipeline.ik.drake_backend import DrakeIKSolver
from src.shared.python.motion_pipeline.ik.mujoco_backend import MuJoCoIKSolver
from src.shared.python.motion_pipeline.ik.opensim_backend import OpenSimIKSolver
from src.shared.python.motion_pipeline.ik.pinocchio_backend import PinocchioIKSolver

from ._local_fixtures import make_3dof_phantom_rig, make_phantom_marker_trajectory

pytestmark = pytest.mark.unit

_STUB_SOLVERS = [
    ("mujoco", MuJoCoIKSolver),
    ("drake", DrakeIKSolver),
    ("opensim", OpenSimIKSolver),
]


def test_pinocchio_backend_raises_import_error_when_unavailable() -> None:
    """With pinocchio mocked spec-less (this tree's conftest), the
    implemented backend reports a missing dependency with an install
    hint — never a silent zero pose."""
    rig = make_3dof_phantom_rig()
    with pytest.raises(ImportError, match="pinocchio"):
        PinocchioIKSolver().solve_frame({}, rig)


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
    ``solve_frame``; otherwise all stubs raise NotImplementedError (#7046).
    """
    rig = make_3dof_phantom_rig()
    traj = make_phantom_marker_trajectory(num_frames=2)
    with pytest.raises((NotImplementedError, ImportError)):
        cls().solve(traj, rig)
