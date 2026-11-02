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
from tests.unit.motion_pipeline.matching_completeness._pinocchio_guard import (
    mock_skip_reason,
)


def _require_real_pinocchio() -> None:
    """Skip if pinocchio is absent or a MagicMock; fail if real but broken.

    ``pytest.importorskip`` alone is insufficient: the ``_protect_engine_modules``
    autouse fixture may restore a ``MagicMock`` for ``pinocchio`` in
    ``sys.modules``. That contamination and a missing library are typed skips.
    A real Pinocchio whose ``Model()`` constructor raises is a provisioning
    defect and raises ``BrokenPinocchioError`` so the module errors at
    collection instead of silently skipping (#14297).
    """
    pin = pytest.importorskip("pinocchio")
    reason = mock_skip_reason(pin)
    if reason is not None:
        pytest.skip(reason, allow_module_level=True)


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


@pytest.mark.unit
def test_synthetic_rig_model_reports_honest_non_production_failure():
    rig = _pendulum_rig()
    traj = _static_traj(rig, q_value=0.0)
    solver = PinocchioInverseDynMatchingSolver()
    result = solver.match(traj, rig)
    assert result.success is False
    assert result.metadata["model_fidelity"] == "synthetic_point_mass"
    assert result.metadata["production_ready"] is False
    assert result.metadata["model_source"] == "synthetic_rig"
    assert "URDF" in (result.message or "")
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
    assert result.success is False


@pytest.mark.unit
def test_urdf_path_must_exist_for_production_solver(tmp_path):
    missing = tmp_path / "missing.urdf"
    with pytest.raises(ValueError, match="URDF path does not exist"):
        PinocchioInverseDynMatchingSolver(urdf_path=missing)


def test_match_rejects_empty_reference():
    rig = _pendulum_rig()
    solver = PinocchioInverseDynMatchingSolver()
    with pytest.raises(ValueError):
        solver.match(None, rig)
