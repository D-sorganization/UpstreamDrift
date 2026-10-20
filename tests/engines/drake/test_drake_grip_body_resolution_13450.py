"""Fit and simulate must resolve the grip body identically (issue #13450).

Drake samples the same canonical frames as Pinocchio: ``mid_hands`` for the
grip and ``club_head`` for the clubhead. Legacy names are aliases of those
frames only; a model without them fails closed instead of silently sampling
``right_hand`` or the shaft origin.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.engines.physics_engines.drake.python.motion_matching import (
    _simulate_support,
    fit_swing_autodiff,
)
from src.engines.physics_engines.drake.python.motion_matching.simulate import (
    CLUBHEAD_FRAME_NAME,
    GRIP_FRAME_NAME,
    SimOptions,
)
from src.engines.physics_engines.pinocchio.python.motion_matching import (
    simulate as pinocchio_simulate,
)

pytestmark = pytest.mark.unit


class _FakePlant:
    def __init__(self, bodies: list[str]) -> None:
        self._bodies = {name: f"body:{name}" for name in bodies}

    def HasBodyNamed(self, name: str) -> bool:  # noqa: N802 - Drake API
        return name in self._bodies

    def GetBodyByName(self, name: str) -> Any:  # noqa: N802 - Drake API
        return self._bodies[name]


_CANONICAL_BODIES = ["right_hand", "left_hand", "mid_hands", "club_shaft", "club_head"]


def test_default_frames_match_pinocchio() -> None:
    assert GRIP_FRAME_NAME == pinocchio_simulate.GRIP_FRAME_NAME == "mid_hands"
    assert CLUBHEAD_FRAME_NAME == pinocchio_simulate.CLUBHEAD_FRAME_NAME
    opts = SimOptions()
    assert opts.grip_body_name == GRIP_FRAME_NAME
    assert opts.clubhead_body_name == CLUBHEAD_FRAME_NAME


def test_defaults_resolve_exactly_on_the_canonical_model() -> None:
    plant = _FakePlant(_CANONICAL_BODIES)
    opts = SimOptions()
    assert _simulate_support._resolve_body_name(plant, opts.grip_body_name) == (
        "mid_hands"
    )
    assert _simulate_support._resolve_body_name(plant, opts.clubhead_body_name) == (
        "club_head"
    )


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("club_grip", "mid_hands"),
        ("grip", "mid_hands"),
        ("clubhead", "club_head"),
    ],
)
def test_legacy_names_alias_the_canonical_frames(alias: str, canonical: str) -> None:
    plant = _FakePlant(_CANONICAL_BODIES)
    assert _simulate_support._resolve_body_name(plant, alias) == canonical
    name, body = fit_swing_autodiff._resolve_grip_body(plant, alias)
    assert name == canonical
    assert body == f"body:{canonical}"


def test_grip_never_falls_back_to_the_hand() -> None:
    """``right_hand`` is the wrist, not the grip: absent frame fails closed."""
    plant = _FakePlant(["club_shaft", "right_hand"])
    assert _simulate_support._resolve_body_name(plant, "club_grip") is None
    assert _simulate_support._resolve_body_name(plant, GRIP_FRAME_NAME) is None
    with pytest.raises(ValueError, match="club_grip"):
        fit_swing_autodiff._resolve_grip_body(plant, "club_grip")


def test_clubhead_never_falls_back_to_the_shaft() -> None:
    plant = _FakePlant(["club_shaft", "right_hand"])
    assert _simulate_support._resolve_body_name(plant, "clubhead") is None
    assert _simulate_support._resolve_body_name(plant, CLUBHEAD_FRAME_NAME) is None


def test_exact_name_wins() -> None:
    plant = _FakePlant(["club_grip", "mid_hands"])
    name, _ = fit_swing_autodiff._resolve_grip_body(plant, "club_grip")
    assert name == "club_grip"


def test_missing_grip_raises() -> None:
    with pytest.raises(ValueError, match="club_grip"):
        fit_swing_autodiff._resolve_grip_body(_FakePlant(["torso"]), "club_grip")
