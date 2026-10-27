"""Bushing-law grip force cap ``force_limit_n`` (HRG-HX4, #13761, option (c)).

Under the option (c) ruling the per-hand grip force limit is enforced only in
the Python bushing law: the force magnitude is capped, its direction is kept,
the moment is untouched, and the default (no limit) is bit-identical to the
uncapped law.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.shared.python.grip_contact.bushing_law import BushingState, bushing_wrench
from src.shared.python.grip_contact.parameters import BushingParameters

pytestmark = pytest.mark.unit

PARAMS = BushingParameters(
    (1.0e6, 9.0e5, 8.0e5),
    (1600.0, 1500.0, 1400.0),
    (554.0, 209.0, 212.0),
    (0.81, 28.6, 27.3),
)


def _state(rotvec, pos, vel=(0, 0, 0), omega=(0, 0, 0)) -> BushingState:
    return BushingState(
        Rotation.from_rotvec(rotvec).as_matrix(),
        np.asarray(pos, float),
        np.asarray(vel, float),
        np.asarray(omega, float),
    )


HAND = _state([0.2, -0.1, 0.3], [0.1, 0.2, 0.3], vel=[0.3, -0.2, 0.1], omega=[1, 2, 3])
CLUB = _state([0.21, -0.09, 0.31], [0.101, 0.198, 0.3005], vel=[0.2, 0.1, 0.0])


def test_force_cap_keeps_direction_and_caps_magnitude() -> None:
    free = bushing_wrench(PARAMS, HAND, CLUB)
    magnitude = float(np.linalg.norm(free.force_n))
    cap = 0.25 * magnitude
    capped = bushing_wrench(PARAMS, HAND, CLUB, force_limit_n=cap)
    assert np.linalg.norm(capped.force_n) == pytest.approx(cap, rel=1e-12)
    np.testing.assert_allclose(
        capped.force_n / cap, free.force_n / magnitude, rtol=1e-12, atol=1e-15
    )
    assert np.array_equal(capped.moment_nm, free.moment_nm)
    assert np.array_equal(capped.angles_rad, free.angles_rad)
    assert np.array_equal(capped.translation_m, free.translation_m)


def test_default_and_slack_cap_leave_the_law_unchanged() -> None:
    free = bushing_wrench(PARAMS, HAND, CLUB)
    explicit_none = bushing_wrench(PARAMS, HAND, CLUB, force_limit_n=None)
    slack = bushing_wrench(
        PARAMS, HAND, CLUB, force_limit_n=2.0 * float(np.linalg.norm(free.force_n))
    )
    for wrench in (explicit_none, slack):
        assert np.array_equal(wrench.force_n, free.force_n)
        assert np.array_equal(wrench.moment_nm, free.moment_nm)


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_force_cap_rejects_non_positive_or_non_finite(bad: float) -> None:
    with pytest.raises(ValueError, match="force_limit_n"):
        bushing_wrench(PARAMS, HAND, CLUB, force_limit_n=bad)
