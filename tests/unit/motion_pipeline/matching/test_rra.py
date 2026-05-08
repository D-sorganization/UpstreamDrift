"""Unit tests for matching.rra (OpenSim RRA)."""

from __future__ import annotations

import pytest

from src.shared.python.motion_pipeline.matching.base import MotionMatchingResult
from src.shared.python.motion_pipeline.matching.rra import RRAMatchingSolver

from ._local_fixtures import make_pendulum_reference_trajectory, make_simple_rig


def test_rra_solver_constructs() -> None:
    assert RRAMatchingSolver() is not None


def test_rra_solver_match_returns_placeholder_failure() -> None:
    s = RRAMatchingSolver()
    ref = make_pendulum_reference_trajectory(num_frames=5)
    rig = make_simple_rig(num_joints=1)
    result = s.match(ref, rig)
    assert isinstance(result, MotionMatchingResult)
    assert result.success is False
    assert "RRA" in (result.message or "")
    assert result.metadata.get("backend") == "rra"


def test_rra_solver_with_opensim() -> None:
    pytest.importorskip("opensim")
    s = RRAMatchingSolver()
    ref = make_pendulum_reference_trajectory(num_frames=5)
    rig = make_simple_rig(num_joints=1)
    result = s.match(ref, rig)
    assert isinstance(result, MotionMatchingResult)
