"""Fixtures for the V&V suite (issue #8616).

The solvers here use :class:`~bunkershot3d.solvers.RefusalPolicy.REPORT`
and, for the analytic cases,
:class:`~bunkershot3d.solvers.ZeroDepression`.  Both choices are about
*verification*, not about physics: a code-verification case asks whether
the arithmetic is right, and a solver that refuses every bunker-relevant
speed -- correctly, because they are all outside the published envelope --
cannot answer that question.  The tests that are about the envelope set
the strict policy explicitly.
"""

from __future__ import annotations

import pytest

from bunkershot3d.sand import PlayingCondition, SandState, playing_condition
from bunkershot3d.solvers import DRFTSolver, MaterialResponse, RefusalPolicy
from bunkershot3d.vandv import quasi_static_solver


@pytest.fixture
def firm_sand() -> SandState:
    """A firm USGA-spec bunker, the default design condition."""
    return playing_condition(PlayingCondition.FIRM)


@pytest.fixture
def material(firm_sand: SandState) -> MaterialResponse:
    """The F0 material response for :func:`firm_sand`."""
    return MaterialResponse.from_sand_state(firm_sand)


@pytest.fixture
def exact_solver(material: MaterialResponse) -> DRFTSolver:
    """A solver with ``delta_h = 0``, so the closed forms are exact."""
    return quasi_static_solver(material)


@pytest.fixture
def default_solver(material: MaterialResponse) -> DRFTSolver:
    """The shipped solver: the default structural correction, reporting."""
    return DRFTSolver(material=material, refusal_policy=RefusalPolicy.REPORT)
