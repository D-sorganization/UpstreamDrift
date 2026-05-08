"""Coverage tests for ``sim_out.FitResult`` postcondition checks."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.motion_matching.cost import CostBreakdown, SimOutput
from src.shared.python.motion_matching.sim_out import FitResult

from ._fixtures import make_target


def _good_simout(n: int = 301) -> SimOutput:
    butt = np.zeros((n, 3))
    clubhead = np.zeros((n, 3))
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    return SimOutput(butt=butt, clubhead=clubhead, club_quat=quat)


def _good_breakdown(total: float = 0.0) -> CostBreakdown:
    return CostBreakdown(
        position=total / 4,
        orientation=total / 4,
        impact_anchor=total / 4,
        regularizer=total / 4,
        total=total,
    )


def test_construct_valid() -> None:
    """Pin: a self-consistent FitResult constructs."""
    fr = FitResult(
        theta=np.zeros(7),
        cost=4.0,
        breakdown=_good_breakdown(4.0),
        target=make_target(),
        sim=_good_simout(),
        engine="drake",
    )
    assert fr.engine == "drake"


def test_non_finite_cost_rejected() -> None:
    """Pin: NaN cost rejected."""
    with pytest.raises(ValueError, match="must be finite"):
        FitResult(
            theta=np.zeros(7),
            cost=float("nan"),
            breakdown=_good_breakdown(),
            target=make_target(),
            sim=_good_simout(),
            engine="drake",
        )


def test_negative_cost_rejected() -> None:
    """Pin: negative cost rejected."""
    with pytest.raises(ValueError, match="non-negative"):
        FitResult(
            theta=np.zeros(7),
            cost=-1.0,
            breakdown=_good_breakdown(-1.0),
            target=make_target(),
            sim=_good_simout(),
            engine="drake",
        )


def test_breakdown_total_must_match_cost() -> None:
    """Pin: breakdown.total must equal cost (within tiny tolerance)."""
    with pytest.raises(ValueError, match="breakdown.total"):
        FitResult(
            theta=np.zeros(7),
            cost=4.0,
            breakdown=_good_breakdown(2.0),
            target=make_target(),
            sim=_good_simout(),
            engine="drake",
        )


def test_engine_must_be_nonempty_str() -> None:
    """Pin: empty engine string rejected."""
    with pytest.raises(ValueError, match="non-empty string"):
        FitResult(
            theta=np.zeros(7),
            cost=0.0,
            breakdown=_good_breakdown(0.0),
            target=make_target(),
            sim=_good_simout(),
            engine="",
        )
