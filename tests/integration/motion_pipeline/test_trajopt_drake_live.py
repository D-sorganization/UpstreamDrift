"""Live-Drake tests for the direct-collocation matching solver.

Dependency-present acceptance tests from #8131 (implemented under epic
#8390, B2/#8397): with pydrake importable, the solver must return
``success=True`` with a non-empty tracked trajectory and finite metrics.
Lives outside ``tests/unit`` because that tree's conftest replaces
``pydrake`` with a spec-less MagicMock.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from src.shared.python.motion_pipeline.contracts import (
    JointDef,
    JointLimit,
    JointStateFrame,
    JointTrajectory,
    SkeletonRig,
)
from src.shared.python.motion_pipeline.matching.base import (
    MatchingBackendType,
    MotionMatchingResult,
    make_matching_solver,
)
from src.shared.python.motion_pipeline.matching.trajopt_drake import (
    DrakeTrajoptMatchingSolver,
)


def _pydrake_available() -> bool:
    try:
        return importlib.util.find_spec("pydrake") is not None
    except (ValueError, ModuleNotFoundError):
        return False


PYDRAKE_AVAILABLE = _pydrake_available()

pytestmark = [
    pytest.mark.integration,
    pytest.mark.motion_pipeline,
    pytest.mark.requires_drake,
    pytest.mark.skipif(not PYDRAKE_AVAILABLE, reason="pydrake not installed"),
]


def _make_simple_rig(num_joints: int = 1) -> SkeletonRig:
    joints = {}
    for i in range(num_joints):
        parent = None if i == 0 else f"j{i - 1}"
        joints[f"j{i}"] = JointDef(
            name=f"j{i}",
            parent=parent,
            children=[f"j{i + 1}"] if i < num_joints - 1 else [],
            tpose_offset=[0.1, 0.0, 0.0],
            axes=["X"],
        )
    return SkeletonRig(id="rig_simple", joints=joints, root_joint="j0")


def _make_pendulum_reference(
    num_frames: int = 5, fps: float = 100.0
) -> JointTrajectory:
    rig = _make_simple_rig(num_joints=1)
    t = np.arange(num_frames) / fps
    q_vals = 0.5 * np.sin(2 * np.pi * t)
    frames = [
        JointStateFrame(timestamp=float(ts), q=[float(q_vals[i])], frame_index=i)
        for i, ts in enumerate(t)
    ]
    return JointTrajectory(id="pendulum_ref", skeleton=rig, frames=frames)


def test_drake_trajopt_with_pydrake() -> None:
    """#8131 acceptance: real solve, real trajectory, finite metrics."""
    s = DrakeTrajoptMatchingSolver()
    ref = _make_pendulum_reference(num_frames=5)
    rig = _make_simple_rig(num_joints=1)
    result = s.match(ref, rig)
    assert isinstance(result, MotionMatchingResult)
    assert result.success is True
    assert result.tracked_trajectory is not None
    assert len(result.tracked_trajectory.frames) == len(ref.frames)
    assert result.fit_metrics
    assert np.isfinite(result.fit_metrics["rmse"])
    assert result.fit_metrics["rmse"] < 0.05
    assert result.metadata.get("production_ready") is True


def test_drake_trajopt_via_production_factory() -> None:
    """The backend is production-exposed (no allow_experimental needed)."""
    solver = make_matching_solver(MatchingBackendType.TRAJOPT_DRAKE)
    ref = _make_pendulum_reference(num_frames=5)
    rig = _make_simple_rig(num_joints=1)
    result = solver.match(ref, rig)
    assert result.success is True


def test_drake_trajopt_tracks_multi_axis_rig() -> None:
    """Multi-axis rig joints expand through the URDF bridge and solve."""
    joints = {
        "j0": JointDef(
            name="j0",
            parent=None,
            children=["j1"],
            tpose_offset=[0.1, 0.0, 0.0],
            axes=["X"],
            limits=[JointLimit(lower=-3.1, upper=3.1)],
        ),
        "j1": JointDef(
            name="j1",
            parent="j0",
            children=[],
            tpose_offset=[0.2, 0.0, 0.0],
            axes=["Y", "Z"],
            limits=[
                JointLimit(lower=-1.5, upper=1.5),
                JointLimit(lower=-1.5, upper=1.5),
            ],
        ),
    }
    rig = SkeletonRig(id="rig3", joints=joints, root_joint="j0")
    t = np.arange(8) / 100.0
    frames = [
        JointStateFrame(
            timestamp=float(ts),
            frame_index=i,
            q=[
                float(0.3 * np.sin(6 * ts)),
                float(0.2 * np.cos(6 * ts) - 0.2),
                float(0.1 * np.sin(3 * ts)),
            ],
        )
        for i, ts in enumerate(t)
    ]
    ref = JointTrajectory(id="ref3", skeleton=rig, frames=frames)
    result = DrakeTrajoptMatchingSolver().match(ref, rig)
    assert result.success is True
    assert result.fit_metrics["rmse"] < 0.05


def test_drake_trajopt_subsamples_long_references() -> None:
    """Captures beyond the knot cap are tracked on a subsampled grid."""
    rig = _make_simple_rig(num_joints=1)
    t = np.arange(120) / 100.0
    frames = [
        JointStateFrame(
            timestamp=float(ts),
            frame_index=i,
            q=[float(0.4 * np.sin(2 * np.pi * ts))],
        )
        for i, ts in enumerate(t)
    ]
    ref = JointTrajectory(id="long_ref", skeleton=rig, frames=frames)
    result = DrakeTrajoptMatchingSolver().match(ref, rig)
    assert result.success is True
    assert result.fit_metrics["num_knots"] == 50
    assert result.fit_metrics["rmse"] < 0.1
