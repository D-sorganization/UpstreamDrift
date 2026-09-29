"""Tests for ``MultiSourceTarget`` (issue #4480).

These tests cover the validation contract of the new multi-source
container.  Because the dependent target types (``ClubBallTarget`` from
#4479 and ``BodyTarget`` from #4476) may not be present on ``main`` yet,
we duck-type the body-side slot via a small in-test stub that mimics the
required ``time`` attribute.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from src.shared.python.motion_matching.multi_source_target import MultiSourceTarget
from tests.unit.motion_matching._fixtures import make_target

pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class _BodyStub:
    """Minimal duck-type for a BodyTarget (only ``time`` is consumed today)."""

    time: np.ndarray


def test_requires_at_least_one_slot() -> None:
    with pytest.raises(ValueError, match="at least one"):
        MultiSourceTarget(club=None, body=None)


def test_club_only_constructs_and_predicates() -> None:
    club = make_target(n=128)
    mst = MultiSourceTarget(club=club, body=None)
    assert mst.has_club() is True
    assert mst.has_body() is False
    assert mst.is_plain_club() is True
    assert mst.is_club_ball() is False


def test_body_only_constructs_and_predicates() -> None:
    t = np.linspace(0.0, 0.3, 64)
    body = _BodyStub(time=t)
    mst = MultiSourceTarget(club=None, body=body)
    assert mst.has_club() is False
    assert mst.has_body() is True
    assert np.array_equal(mst.shared_time(), t)


def test_shared_time_returns_club_when_both_present() -> None:
    club = make_target(n=64)
    body = _BodyStub(time=club.time.copy())
    mst = MultiSourceTarget(club=club, body=body)
    assert np.array_equal(mst.shared_time(), club.time)


def test_mismatched_timegrid_raises() -> None:
    club = make_target(n=64)
    body = _BodyStub(time=np.linspace(0.0, 0.3, 65))  # off-by-one length
    with pytest.raises(ValueError, match="timegrid mismatch"):
        MultiSourceTarget(club=club, body=body)


def test_mismatched_timegrid_values_raises() -> None:
    club = make_target(n=64)
    # same shape but offset values → still a mismatch
    body = _BodyStub(time=club.time + 1.0e-3)
    with pytest.raises(ValueError, match="timegrid mismatch"):
        MultiSourceTarget(club=club, body=body)


def test_type_guard_on_club_slot() -> None:
    with pytest.raises(TypeError, match="time"):
        MultiSourceTarget(club="not a target", body=_BodyStub(time=np.zeros(8)))


def test_type_guard_on_body_slot() -> None:
    with pytest.raises(TypeError, match="time"):
        MultiSourceTarget(club=make_target(n=8), body=42)


def test_is_club_ball_detected_via_ball_attribute() -> None:
    """A ClubBallTarget exposes a ``.ball`` attribute composing the impact.

    We stub one here so the predicate can be exercised before the real
    type lands (#4479).
    """
    plain = make_target(n=32)

    @dataclass(frozen=True)
    class _ClubBallStub:
        time: np.ndarray
        club: object  # would be ClubTarget once #4479 lands
        ball: object  # presence of this attribute is the duck-type

    cb = _ClubBallStub(time=plain.time, club=plain, ball=object())
    mst = MultiSourceTarget(club=cb, body=None)
    assert mst.is_club_ball() is True
    assert mst.is_plain_club() is False
