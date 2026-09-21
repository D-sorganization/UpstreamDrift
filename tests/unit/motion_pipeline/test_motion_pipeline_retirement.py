"""Unit tests for retirement of legacy CIR solvers per ADR-0051 (MS-12 / #10331).

Verifies:
1. make_ik_solver raises actionable NotImplementedError referencing ADR-0051
   for retired stubs: opensim, mujoco, drake.
2. make_matching_solver raises actionable ValueError referencing ADR-0051
   for retired solvers: cmc, rra.
3. Active production solvers remain available.
"""

from __future__ import annotations

import pytest

from src.shared.python.motion_pipeline.ik.base import (
    IKBackendType,
    make_ik_solver,
)
from src.shared.python.motion_pipeline.matching.base import (
    MatchingBackendType,
    make_matching_solver,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "backend",
    [
        IKBackendType.MUJOCO,
        IKBackendType.OPENSIM,
        IKBackendType.DRAKE,
        "mujoco",
        "opensim",
        "drake",
    ],
)
def test_retired_ik_solvers_raise_adr0051(backend: IKBackendType | str) -> None:
    """Attempting to instantiate retired IK backends must raise with ADR-0051 reference."""
    with pytest.raises(NotImplementedError, match="ADR-0051"):
        make_ik_solver(backend)


@pytest.mark.unit
@pytest.mark.parametrize(
    "backend",
    [
        MatchingBackendType.CMC,
        MatchingBackendType.RRA,
        "cmc",
        "rra",
    ],
)
def test_retired_matching_solvers_raise_adr0051(
    backend: MatchingBackendType | str,
) -> None:
    """Attempting to create retired matching solvers must raise with ADR-0051 reference."""
    with pytest.raises(ValueError, match="ADR-0051"):
        make_matching_solver(backend)


@pytest.mark.unit
def test_active_ik_solvers_remain_callable() -> None:
    """Geometric and Pinocchio IK solvers remain available."""
    geo = make_ik_solver(IKBackendType.GEOMETRIC)
    assert hasattr(geo, "solve")

    pin = make_ik_solver(IKBackendType.PINOCCHIO)
    assert hasattr(pin, "solve")


@pytest.mark.unit
def test_active_matching_solvers_remain_callable() -> None:
    """Active production matching solvers remain available."""
    drake = make_matching_solver(MatchingBackendType.TRAJOPT_DRAKE)
    assert hasattr(drake, "match")
