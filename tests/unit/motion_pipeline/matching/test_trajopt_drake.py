"""Unit tests for matching.trajopt_drake (epic #8390, B2/#8397)."""

from __future__ import annotations

from unittest.mock import patch


from src.shared.python.motion_pipeline.matching.base import MotionMatchingResult
from src.shared.python.motion_pipeline.matching.trajopt_drake import (
    DrakeTrajoptMatchingSolver,
)

from ._local_fixtures import make_pendulum_reference_trajectory, make_simple_rig


def test_drake_trajopt_solver_constructs() -> None:
    assert DrakeTrajoptMatchingSolver() is not None


def test_drake_trajopt_without_pydrake_reports_dependency_missing() -> None:
    """Absence of pydrake must degrade to a failed result with an install
    hint — never an exception (guarded optional dependency policy)."""
    with patch(
        "src.shared.python.motion_pipeline.matching.trajopt_drake._drake_available",
        return_value=False,
    ):
        s = DrakeTrajoptMatchingSolver()
        ref = make_pendulum_reference_trajectory(num_frames=5)
        rig = make_simple_rig(num_joints=1)
        result = s.match(ref, rig)
    assert result.success is False
    assert "pydrake" in result.message
    assert result.metadata.get("status") == "dependency_missing"


def test_drake_trajopt_under_mocked_pydrake_degrades_cleanly() -> None:
    """tests/unit's conftest installs a spec-less pydrake MagicMock; the
    availability probe must treat that as unavailable (not raise ValueError).

    The real dependency-present solves live in
    tests/integration/motion_pipeline/test_trajopt_drake_live.py where
    pydrake is not mocked (the #8131 acceptance test moved there when the
    solver was implemented — epic #8390, B2/#8397).
    """
    s = DrakeTrajoptMatchingSolver()
    ref = make_pendulum_reference_trajectory(num_frames=5)
    rig = make_simple_rig(num_joints=1)
    result = s.match(ref, rig)
    assert isinstance(result, MotionMatchingResult)
    # Under the unit-conftest mock the backend reports dependency_missing;
    # in an environment with importable real pydrake it solves. Both are
    # valid here — what is forbidden is an exception.
    assert result.metadata.get("backend") == "drake_trajopt"
