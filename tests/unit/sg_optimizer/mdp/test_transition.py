"""Isolated tests for hazard rules and ``sample_transitions`` DbC guards.

Covers issue #7715: the scoring-critical water/OB stroke-penalty drop in
``_apply_hazard_rules`` and the two ``require()`` preconditions on
``sample_transitions`` were never exercised directly. The water/OB +1-stroke
drop is the heart of strokes-gained correctness, so a regression that dropped
the penalty or mis-mapped a lie code would silently corrupt every result while
passing the coarse integration checks.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.sg_optimizer.course.rasterize import LIE_CODES
from src.shared.python.sg_optimizer.mdp.state import State
from src.shared.python.sg_optimizer.mdp.transition import (
    _apply_hazard_rules,
    sample_transitions,
)

pytestmark = pytest.mark.unit

_FAIRWAY = LIE_CODES["fairway"]
_ROUGH = LIE_CODES["rough"]
_WATER = LIE_CODES["water"]
_OB = LIE_CODES["ob"]
_HOLED = LIE_CODES["holed"]


def _origin() -> State:
    return State(x=10.0, y=20.0, lie=_ROUGH)


@pytest.mark.parametrize("hazard", [_WATER, _OB])
def test_hazard_drop_adds_one_stroke_and_returns_to_origin(hazard: int) -> None:
    """Water/OB landings yield extra_strokes==1 and next_state == origin."""
    origin = _origin()
    outcome = _apply_hazard_rules(origin, x=55.5, y=-12.0, landing_lie=hazard)

    assert outcome.extra_strokes == 1
    # The drop returns the ball to the originating position and lie, not the
    # hazard landing coordinates.
    assert outcome.next_state.x == origin.x
    assert outcome.next_state.y == origin.y
    assert outcome.next_state.lie == origin.lie


def test_non_hazard_landing_has_no_penalty_at_landing() -> None:
    """A fairway landing yields extra_strokes==0 at the landing position."""
    origin = _origin()
    outcome = _apply_hazard_rules(origin, x=55.5, y=-12.0, landing_lie=_FAIRWAY)

    assert outcome.extra_strokes == 0
    assert outcome.next_state.x == 55.5
    assert outcome.next_state.y == -12.0
    assert outcome.next_state.lie == _FAIRWAY


def test_sample_transitions_rejects_non_positive_n_samples() -> None:
    """The require(n_samples > 0) precondition raises (ValueError subclass)."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        sample_transitions(
            state=_origin(),
            action=object(),  # type: ignore[arg-type]
            profile=object(),  # type: ignore[arg-type]
            baseline=object(),  # type: ignore[arg-type]
            conditions=object(),  # type: ignore[arg-type]
            raster=object(),  # type: ignore[arg-type]
            n_samples=0,
            rng=rng,
        )


def test_sample_transitions_rejects_action_from_holed_state() -> None:
    """The require(state.lie != HOLED) precondition raises (ValueError subclass)."""
    rng = np.random.default_rng(0)
    holed = State(x=0.0, y=0.0, lie=_HOLED)
    with pytest.raises(ValueError):
        sample_transitions(
            state=holed,
            action=object(),  # type: ignore[arg-type]
            profile=object(),  # type: ignore[arg-type]
            baseline=object(),  # type: ignore[arg-type]
            conditions=object(),  # type: ignore[arg-type]
            raster=object(),  # type: ignore[arg-type]
            n_samples=4,
            rng=rng,
        )
