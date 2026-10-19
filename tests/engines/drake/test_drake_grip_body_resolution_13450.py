"""Fit and simulate must resolve the grip body identically (issue #13450)."""

from __future__ import annotations

from typing import Any

import pytest

from src.engines.physics_engines.drake.python.motion_matching import (
    _simulate_support,
    fit_swing_autodiff,
)


pytestmark = pytest.mark.unit


class _FakePlant:
    def __init__(self, bodies: list[str]) -> None:
        self._bodies = {name: f"body:{name}" for name in bodies}

    def HasBodyNamed(self, name: str) -> bool:  # noqa: N802 - Drake API
        return name in self._bodies

    def GetBodyByName(self, name: str) -> Any:  # noqa: N802 - Drake API
        return self._bodies[name]


def test_fit_uses_same_fallback_as_simulate() -> None:
    plant = _FakePlant(["club_shaft", "right_hand"])  # no club_grip
    simulate_name = _simulate_support._resolve_body_name(plant, "club_grip")
    assert simulate_name == "right_hand"
    name, body = fit_swing_autodiff._resolve_grip_body(plant, "club_grip")
    assert name == simulate_name
    assert body == "body:right_hand"


def test_exact_name_wins() -> None:
    plant = _FakePlant(["club_grip", "right_hand"])
    name, _ = fit_swing_autodiff._resolve_grip_body(plant, "club_grip")
    assert name == "club_grip"


def test_missing_grip_raises() -> None:
    with pytest.raises(ValueError, match="club_grip"):
        fit_swing_autodiff._resolve_grip_body(_FakePlant(["torso"]), "club_grip")
