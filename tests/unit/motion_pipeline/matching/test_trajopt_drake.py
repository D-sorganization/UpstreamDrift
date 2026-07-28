"""Unit tests for matching.trajopt_drake."""

from __future__ import annotations

import pytest

from src.shared.python.motion_pipeline.matching.base import MotionMatchingResult
from src.shared.python.motion_pipeline.matching.trajopt_drake import (
    DrakeTrajoptMatchingSolver,
)

from ._local_fixtures import make_pendulum_reference_trajectory, make_simple_rig


def test_drake_trajopt_solver_constructs() -> None:
    assert DrakeTrajoptMatchingSolver() is not None


@pytest.mark.xfail(
    reason="Drake trajectory optimization matching is not implemented yet; see #8131",
    strict=True,
)
def test_drake_trajopt_with_pydrake() -> None:
    pytest.importorskip("pydrake")
    s = DrakeTrajoptMatchingSolver()
    ref = make_pendulum_reference_trajectory(num_frames=5)
    rig = make_simple_rig(num_joints=1)
    result = s.match(ref, rig)
    assert isinstance(result, MotionMatchingResult)
    assert result.success is True
    assert result.tracked_trajectory is not None
    assert result.fit_metrics
    assert result.metadata.get("production_ready") is True
