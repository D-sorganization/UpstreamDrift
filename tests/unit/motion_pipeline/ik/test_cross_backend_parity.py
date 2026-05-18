"""Cross-backend parity test for IK (#4566 headline spec).

Compares joint-angle output of all four IK backends on a 3-DOF phantom.
Each backend is currently a placeholder returning the neutral pose, so
parity is trivially satisfied; once any backend lands a real
implementation, this test will catch divergence.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.motion_pipeline.ik.drake_backend import DrakeIKSolver
from src.shared.python.motion_pipeline.ik.mujoco_backend import MuJoCoIKSolver
from src.shared.python.motion_pipeline.ik.opensim_backend import OpenSimIKSolver
from src.shared.python.motion_pipeline.ik.pinocchio_backend import PinocchioIKSolver

from ._local_fixtures import make_3dof_phantom_rig, make_phantom_marker_trajectory

_TOL_RAD = math.radians(5.0)  # 5° tolerance per spec


def _available_backends() -> list[tuple[str, object]]:
    """Return placeholders + pytest.importorskip-aware list."""
    backends: list[tuple[str, object]] = [
        ("mujoco", MuJoCoIKSolver()),
        ("drake", DrakeIKSolver()),
        ("pinocchio", PinocchioIKSolver()),
    ]
    # OpenSim's solve() raises ImportError without the package; only include
    # it when import succeeds.
    try:
        import opensim  # noqa: F401

        backends.append(("opensim", OpenSimIKSolver()))
    except ImportError:
        pass
    return backends


def test_cross_backend_parity_on_3dof_phantom() -> None:
    rig = make_3dof_phantom_rig()
    traj = make_phantom_marker_trajectory(num_frames=5)

    available = _available_backends()
    if len(available) < 2:
        pytest.xfail("Cross-backend parity needs >= 2 backends available")

    results: dict[str, np.ndarray] = {}
    for name, solver in available:
        out = solver.solve(traj, rig)
        q_arr = np.array([f.q for f in out.frames])
        results[name] = q_arr

    backend_names = list(results.keys())
    reference = results[backend_names[0]]
    for name in backend_names[1:]:
        diff = np.abs(results[name] - reference)
        assert diff.max() <= _TOL_RAD, (
            f"{name} differs from {backend_names[0]} by {diff.max()} rad > 5°"
        )
