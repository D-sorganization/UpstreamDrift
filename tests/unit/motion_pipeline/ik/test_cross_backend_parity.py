"""Parity / determinism test for the real IK backend (#4566, #7046).

The engine-specific backends (mujoco/drake/opensim/pinocchio) are
unimplemented stubs that raise ``NotImplementedError`` (issue #7046), so a
cross-backend numerical comparison is no longer meaningful. Instead we pin
the determinism of the one real backend - geometric - which is the
reference all future backends must agree with.
"""

from __future__ import annotations

import numpy as np

from src.shared.python.motion_pipeline.contracts import (
    Marker,
    MarkerFrame,
    MarkerTrajectory,
)
from src.shared.python.motion_pipeline.ik.geometric_backend import (
    GeometricIKSolver,
    forward_kinematics,
)

from ._local_fixtures import make_3dof_phantom_rig


def _markers_traj_from_rig(rig, num_frames: int = 5) -> MarkerTrajectory:
    """Build a marker trajectory whose markers sit on the rig's joints."""
    frames = []
    for i in range(num_frames):
        t = i / 100.0
        # Small per-frame perturbation on the first DOF.
        q = [0.1 * i] + [0.0] * (rig.num_dofs - 1)
        pos = forward_kinematics(rig, q)
        markers = {
            name: Marker(name=name, x=p[0], y=p[1], z=p[2]) for name, p in pos.items()
        }
        frames.append(MarkerFrame(timestamp=t, markers=markers, frame_index=i))
    return MarkerTrajectory(id="det_traj", frames=frames)


def test_geometric_backend_is_deterministic() -> None:
    rig = make_3dof_phantom_rig()
    traj = _markers_traj_from_rig(rig, num_frames=5)

    out_a = GeometricIKSolver().solve(traj, rig)
    out_b = GeometricIKSolver().solve(traj, rig)

    q_a = np.array([f.q for f in out_a.frames])
    q_b = np.array([f.q for f in out_b.frames])
    assert np.allclose(q_a, q_b), "Geometric IK must be deterministic"
    assert np.all(np.isfinite(q_a))
