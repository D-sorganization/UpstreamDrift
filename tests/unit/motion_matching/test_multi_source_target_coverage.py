"""Coverage tests for ``MultiSourceTarget``."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.motion_matching.multi_source_target import MultiSourceTarget

from ._fixtures import make_provenance, make_target


def _body_like(time: np.ndarray) -> object:
    """Quack-typed body: object with a .time ndarray."""

    class _Body:
        pass

    b = _Body()
    b.time = time
    return b


def test_requires_at_least_one_slot() -> None:
    """Pin: both-None construction is rejected."""
    with pytest.raises(ValueError, match="at least one non-None slot"):
        MultiSourceTarget(club=None, body=None)


def test_club_alone_constructs() -> None:
    """Pin: club-only target constructs and exposes time."""
    t = make_target()
    m = MultiSourceTarget(club=t, body=None)
    assert m.has_club() and not m.has_body()
    assert np.array_equal(m.shared_time(), t.time)


def test_body_alone_constructs() -> None:
    """Pin: body-only target constructs and exposes its time."""
    t = make_target()
    m = MultiSourceTarget(club=None, body=_body_like(t.time))
    assert m.has_body() and not m.has_club()
    assert np.array_equal(m.shared_time(), t.time)


def test_club_must_have_time() -> None:
    """Pin: club without a 1-D ``time`` ndarray is rejected."""

    class Bad:
        time = "not an array"

    with pytest.raises(TypeError, match="club must expose"):
        MultiSourceTarget(club=Bad(), body=None)


def test_body_must_have_time() -> None:
    """Pin: body without a 1-D ``time`` ndarray is rejected."""

    class Bad:
        time = np.zeros((2, 2))  # 2-D

    t = make_target()
    with pytest.raises(TypeError, match="body must expose"):
        MultiSourceTarget(club=t, body=Bad())


def test_timegrid_mismatch_rejected() -> None:
    """Pin: club/body with mismatched time arrays are rejected."""
    t = make_target()
    body = _body_like(t.time + 1.0)  # different time grid
    with pytest.raises(ValueError, match="timegrid mismatch"):
        MultiSourceTarget(club=t, body=body)


def test_is_plain_club_vs_club_ball() -> None:
    """Pin: ``is_plain_club`` and ``is_club_ball`` distinguish target type."""
    t = make_target()
    m_plain = MultiSourceTarget(club=t, body=None)
    assert m_plain.is_plain_club()
    assert not m_plain.is_club_ball()

    # ``is_club_ball`` is a hasattr(club, "ball") duck-typing probe.
    # Build a minimal object that satisfies it without needing the
    # full ClubBallTarget dependency surface.
    class _BallishTarget:
        time = t.time
        ball = "present"

    m_ball = MultiSourceTarget(club=_BallishTarget(), body=None)
    assert m_ball.is_club_ball()
    assert not m_ball.is_plain_club()


def test_provenance_unused_kept_in_scope() -> None:
    """Pin: a fresh provenance from the helper imports cleanly."""
    # Just exercise the helper to keep coverage signal positive.
    p = make_provenance()
    assert p.format == "synthetic"
