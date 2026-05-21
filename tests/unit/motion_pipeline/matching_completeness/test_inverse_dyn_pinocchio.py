"""Tests for the Pinocchio RNEA inverse-dynamics motion-matching backend."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_pipeline.contracts import (
    JointDef,
    JointStateFrame,
    JointTrajectory,
    SkeletonRig,
)
from src.shared.python.motion_pipeline.matching.inverse_dyn_pinocchio import (
    PinocchioInverseDynMatchingSolver,
)


def _require_real_pinocchio() -> None:
    """Skip the module if pinocchio is absent or replaced with a MagicMock.

    ``pytest.importorskip`` alone is insufficient: the ``_protect_engine_modules``
    autouse fixture may restore a ``MagicMock`` for ``pinocchio`` in ``sys.modules``
    (from a prior test that mocked the engine probe).  When that happens
    ``importorskip`` happily returns the mock — and subsequent ``pin.Model().nq``
    calls return nested ``MagicMock`` objects rather than integers, causing the
    solver's DOF-consistency check to raise ``ValueError``.

    This helper additionally verifies that the imported module is the real
    Pinocchio C extension by checking that ``pin.Model().nq`` is an ``int``.
    """
    pin = pytest.importorskip("pinocchio")
    try:
        model = pin.Model()
        if not isinstance(model.nq, int):
            pytest.skip(
                "pinocchio in sys.modules is a MagicMock — real library absent",
                allow_module_level=True,
            )
    except Exception as exc:
        pytest.skip(
            f"pinocchio appears to be mocked or broken: {exc}",
            allow_module_level=True,
        )


_require_real_pinocchio()


def _pendulum_rig() -> SkeletonRig:
    """Single-DoF revolute about X — like a 1-DoF pendulum."""
    return SkeletonRig(
        id="pendulum",
        joints={
            "hinge": JointDef(
                name="hinge",
                parent=None,
                children=[],
                tpose_offset=[0.0, 0.0, -1.0],
                axes=["X"],
            ),
        },
        root_joint="hinge",
    )


def _static_traj(rig: SkeletonRig, q_value: float = 0.0) -> JointTrajectory:
    times = np.linspace(0.0, 0.1, 11)
    frames = [
        JointStateFrame(
            timestamp=float(t),
            q=[q_value],
            qdot=[0.0],
            qddot=[0.0],
            frame_index=i,
        )
        for i, t in enumerate(times)
    ]
    return JointTrajectory(id="ref", skeleton=rig, frames=frames)


def test_match_returns_finite_torques_for_static_pose():
    rig = _pendulum_rig()
    traj = _static_traj(rig, q_value=0.0)
    solver = PinocchioInverseDynMatchingSolver()
    result = solver.match(traj, rig)
    assert result.success
    assert result.torque_trajectory is not None
    assert result.torque_trajectory.num_frames == traj.num_frames
    for f in result.torque_trajectory.frames:
        assert all(np.isfinite(v) for v in f.tau)


def test_match_postconditions():
    rig = _pendulum_rig()
    traj = _static_traj(rig, q_value=0.1)
    solver = PinocchioInverseDynMatchingSolver()
    result = solver.match(traj, rig)
    # tracked_trajectory == reference (kinematic IK pass-through)
    assert result.tracked_trajectory is traj
    assert result.solve_time is not None and result.solve_time >= 0
    assert "rmse" in result.fit_metrics


def test_match_rejects_empty_reference():
    rig = _pendulum_rig()
    solver = PinocchioInverseDynMatchingSolver()
    with pytest.raises(ValueError):
        solver.match(None, rig)
