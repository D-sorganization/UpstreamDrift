"""Shared lead/trail side resolution (HRG-G1, #13186)."""

from __future__ import annotations

import pytest

from src.shared.python.core.handedness import (
    DEFAULT_HANDEDNESS,
    lead_side,
    trail_side,
    validate_handedness,
)

pytestmark = pytest.mark.unit


def test_right_hander_leads_with_the_left_hand() -> None:
    assert (lead_side("right"), trail_side("right")) == ("L", "R")


def test_left_hander_is_mirrored() -> None:
    assert (lead_side("left"), trail_side("left")) == ("R", "L")


def test_default_is_right() -> None:
    assert DEFAULT_HANDEDNESS == "right"


@pytest.mark.parametrize("bad", ["both", "", None, "Right"])
def test_invalid_handedness_raises(bad) -> None:
    with pytest.raises(ValueError, match="handedness"):
        validate_handedness(bad)
    with pytest.raises(ValueError):
        lead_side(bad)
