"""Off-centre strike speed retention for batch putt models (#14461)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.putting_dynamics import (
    PutterState,
    off_centre_speed_retention,
    strike,
)

pytestmark = pytest.mark.unit

_SQUARE = PutterState(head_mass_kg=0.35, moi_kg_m2=4.5e-4, loft_deg=0.0, speed_mps=2.0)


def test_a_centred_strike_keeps_all_its_speed() -> None:
    assert off_centre_speed_retention(_SQUARE, np.zeros(3)) == pytest.approx(1.0)


def test_retention_matches_the_collision_model_for_a_square_face() -> None:
    toe = np.array([-0.015, -0.005, 0.005, 0.010, 0.020])
    centre = strike(_SQUARE).ball_speed_mps
    want = [strike(_SQUARE, float(d)).ball_speed_mps / centre for d in toe]
    np.testing.assert_allclose(off_centre_speed_retention(_SQUARE, toe), want)


def test_heel_and_toe_misses_cost_the_same_and_grow_with_the_miss() -> None:
    toe = np.array([0.0, 0.005, 0.010, 0.015])
    toe_side = off_centre_speed_retention(_SQUARE, toe)
    np.testing.assert_allclose(off_centre_speed_retention(_SQUARE, -toe), toe_side)
    assert np.all(np.diff(toe_side) < 0.0)
    assert toe_side[-1] > 0.95  # a 15 mm miss loses a couple of percent


def test_strikes_off_the_face_are_refused() -> None:
    with pytest.raises(ValueError, match="0.08"):
        off_centre_speed_retention(_SQUARE, np.array([0.0, 0.09]))
    with pytest.raises(ValueError):
        off_centre_speed_retention(_SQUARE, np.array([np.nan]))
