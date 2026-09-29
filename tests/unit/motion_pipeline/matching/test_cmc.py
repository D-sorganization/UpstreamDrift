"""Unit tests for matching.cmc (OpenSim CMC)."""

from __future__ import annotations

import pytest

from src.shared.python.motion_pipeline.matching.base import MotionMatchingResult
from src.shared.python.motion_pipeline.matching.cmc import CMCMatchingSolver

from ._local_fixtures import make_pendulum_reference_trajectory, make_simple_rig


def test_cmc_solver_constructs_without_opensim() -> None:
    s = CMCMatchingSolver()
    assert s is not None


@pytest.mark.xfail(
    reason="CMC muscle redundancy solve is not implemented yet; see #8131",
    strict=True,
)
def test_cmc_solver_with_opensim() -> None:
    pytest.importorskip("opensim")
    s = CMCMatchingSolver()
    ref = make_pendulum_reference_trajectory(num_frames=5)
    rig = make_simple_rig(num_joints=1)
    result = s.match(ref, rig)
    assert isinstance(result, MotionMatchingResult)
    assert result.success is True
    assert (
        result.activation_trajectory is not None or result.torque_trajectory is not None
    )
    assert result.fit_metrics
    assert result.metadata.get("production_ready") is True
