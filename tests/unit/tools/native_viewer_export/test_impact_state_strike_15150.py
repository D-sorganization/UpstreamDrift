"""The contact state is never fitted across the recorded strike (#15150).

A captured swing records its strike at a sample that already carries it. A
ball moved forward meets the face after that sample, and a fit whose stencil
reaches it reads the struck head: the club path stepped 3.5 deg on a 1 cm
ball shift. The stencil ends before the recorded strike sample instead.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.impact_parameters import ClubheadSeries
from src.tools.native_viewer_export import impact_card, impact_state

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

RATE_HZ = 360.0
RADIUS_M = 0.021335
ARC_RADIUS_M = 1.2
TURN_RAD_S = 42.0
STRIKE_INDEX = 180  # first recorded sample that carries the strike
STRIKE_S = STRIKE_INDEX / RATE_HZ
STRUCK_FROM_S = STRIKE_S - 0.5 / RATE_HZ  # the strike lies between samples
CONTACT_S = STRIKE_S + 0.3 / RATE_HZ  # a ball moved forward meets the face here
SLOWDOWN = 0.6
#: a cubic of six samples carried 1.3 steps along the arc reads 0.57 deg off;
#: the struck stencil reads 6.6 deg off
CLEAN_DEG = 1.0
STRUCK_DEG = 3.0


def _arc(t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    turn = TURN_RAD_S * (t - CONTACT_S)
    centre = ARC_RADIUS_M * np.array([math.sin(turn), -math.cos(turn), 0.0])
    velocity = ARC_RADIUS_M * TURN_RAD_S * np.array([math.cos(turn), math.sin(turn), 0])
    return centre, velocity, np.array([math.cos(turn), math.sin(turn), 0.0])


def _series(strike_time_s: float | None = None) -> ClubheadSeries:
    times = np.arange(0.0, 1.0, 1.0 / RATE_HZ)
    lagged = np.where(
        times > STRUCK_FROM_S, STRUCK_FROM_S + SLOWDOWN * (times - STRUCK_FROM_S), times
    )
    states = [_arc(float(t)) for t in lagged]
    centre = np.array([s[0] for s in states])
    normal = np.array([s[2] for s in states])
    return impact_card.ModelClubSeries(
        times_s=times,
        face_center_m=centre,
        velocity_mps=np.gradient(centre, times, axis=0),
        face_normal=normal,
        toe_axis=np.cross(normal, [0.0, 0.0, 1.0]),
        grip_axis=np.tile([0.0, 0.0, 1.0], (len(times), 1)),
        strike_time_s=strike_time_s,
    )


def _ball() -> np.ndarray:
    at, _, face = _arc(CONTACT_S)
    return at + RADIUS_M * face


def _path_error_deg(series: ClubheadSeries, **kwargs: float) -> float:
    found = impact_state.series_at_contact(
        series, _ball(), RADIUS_M, STRIKE_S, **kwargs
    )
    assert found is not None
    moved, index = found
    got = moved.velocity_mps[index]
    want = _arc(CONTACT_S)[1]
    cosine = float(got @ want) / float(np.linalg.norm(got) * np.linalg.norm(want))
    return math.degrees(math.acos(min(cosine, 1.0)))


def test_a_stencil_across_the_strike_reads_the_struck_head() -> None:
    assert _path_error_deg(_series()) > STRUCK_DEG


def test_the_stencil_ends_before_the_recorded_strike() -> None:
    assert _path_error_deg(_series(), strike_time_s=STRIKE_S) < CLEAN_DEG


def test_the_model_series_carries_its_strike_time() -> None:
    assert _path_error_deg(_series(STRIKE_S)) < CLEAN_DEG


def test_the_fitted_samples_all_precede_the_strike() -> None:
    found = impact_state.series_at_contact(
        _series(), _ball(), RADIUS_M, STRIKE_S, strike_time_s=STRIKE_S
    )
    assert found is not None
    moved, index = found
    # postcondition: samples before the contact state are the source samples
    assert np.array_equal(moved.times_s[:index], _series().times_s[:index])
    assert moved.times_s[index] == pytest.approx(CONTACT_S, abs=0.1 / RATE_HZ)


def test_a_strike_before_every_sample_gives_no_contact_state() -> None:
    assert (
        impact_state.series_at_contact(
            _series(), _ball(), RADIUS_M, STRIKE_S, strike_time_s=-1.0
        )
        is None
    )


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_a_non_finite_strike_time_is_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="strike_time_s"):
        impact_state.series_at_contact(
            _series(), _ball(), RADIUS_M, STRIKE_S, strike_time_s=bad
        )
