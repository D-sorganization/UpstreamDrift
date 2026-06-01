"""Cross-engine unification tests for shared provider helpers (#6935).

Before #6935 each engine provider re-implemented target-unwrapping and
leaderboard publication, and the copies had FORKED:

* a ``ClubBallTarget`` unwrapped on Drake/OpenSim/MyoSuite but raised
  ``TypeError`` on MuJoCo/Pendulum/Pinocchio;
* only Pinocchio forwarded ``target_id`` to the leaderboard.

These tests pin the UNIFIED behaviour of the single source of truth
(:func:`resolve_club_target`, :func:`publish_leaderboard_row`). They need no
engine wheels, so they run on every CI host.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_matching.club_ball_target import (
    ClubBallTarget,
    extract_ball_impact_from_clubtarget,
)
from src.shared.python.motion_matching.club_target import (
    ClubTarget,
    SourceProvenance,
)
from src.shared.python.motion_matching.provider import (
    MultiSourceTarget,
    publish_leaderboard_row,
    resolve_club_target,
)


def _make_club_target() -> ClubTarget:
    """Build a tiny valid ``ClubTarget`` for unwrap tests."""
    n = 4
    time = np.linspace(0.0, 0.3, n)
    butt = np.tile(np.array([0.1, 0.2, 0.3]), (n, 1))
    clubhead = np.tile(np.array([0.4, 0.1, 0.2]), (n, 1))
    club_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    source = SourceProvenance(
        filename="synthetic.csv",
        format="synthetic",
        subject_id="T",
        trial_id="trial-42",
        sha256="0" * 64,
    )
    return ClubTarget(
        time=time,
        butt=butt,
        clubhead=clubhead,
        club_quat=club_quat,
        impact_idx=n,
        source=source,
    )


@pytest.fixture
def club() -> ClubTarget:
    return _make_club_target()


@pytest.fixture
def wrappers(club: ClubTarget) -> dict[str, object]:
    """All three supported target wrapper shapes around the same club."""
    ball = extract_ball_impact_from_clubtarget(club)
    return {
        "bare": club,
        "club_ball": ClubBallTarget(club=club, ball_impact=ball),
        "multi_source": MultiSourceTarget(club=club),
    }


@pytest.mark.parametrize("kind", ["bare", "club_ball", "multi_source"])
def test_resolve_club_target_unwraps_uniformly(
    wrappers: dict[str, object], club: ClubTarget, kind: str
) -> None:
    """Every wrapper type resolves to the SAME underlying ClubTarget.

    This is the core #6935 unification: ``ClubBallTarget`` no longer raises
    ``TypeError`` on a subset of engines -- the single shared unwrap accepts
    all three shapes identically.
    """
    resolved = resolve_club_target(wrappers[kind])
    assert resolved is club


def test_resolve_club_target_rejects_empty_multi_source() -> None:
    bad = MultiSourceTarget(body={"present": True})  # club is None
    with pytest.raises(ValueError, match="club"):
        resolve_club_target(bad)


def test_resolve_club_target_rejects_unknown_type() -> None:
    with pytest.raises(TypeError, match="ClubTarget"):
        resolve_club_target(object())


def test_publish_leaderboard_row_noop_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publication is a safe no-op unless UD_LEADERBOARD_PUBLISH=1."""
    monkeypatch.delenv("UD_LEADERBOARD_PUBLISH", raising=False)
    calls: list[tuple] = []

    import src.shared.python.motion_matching.leaderboard as lb

    monkeypatch.setattr(
        lb,
        "append_row",
        lambda *a, **k: calls.append((a, k)),  # type: ignore[misc]
    )
    # Should not raise and should not append (env gate is off).
    publish_leaderboard_row("drake", object(), "1.2.3", target_id="trial-42")
    assert calls == []


def test_publish_leaderboard_row_forwards_target_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When enabled, target_id reaches the leaderboard for ALL engines.

    Previously only Pinocchio forwarded ``target_id``; the shared helper
    forwards it uniformly.
    """
    monkeypatch.setenv("UD_LEADERBOARD_PUBLISH", "1")
    captured: dict[str, object] = {}

    import src.shared.python.motion_matching.leaderboard as lb

    def _fake_append_row(
        engine, fit_result, engine_version, *, json_path=None, target_id=None
    ):
        captured["engine"] = engine
        captured["version"] = engine_version
        captured["target_id"] = target_id
        return json_path

    monkeypatch.setattr(lb, "append_row", _fake_append_row)
    publish_leaderboard_row("mujoco", object(), "3.1.0", target_id="trial-42")
    assert captured["engine"] == "mujoco"
    assert captured["version"] == "3.1.0"
    assert captured["target_id"] == "trial-42"
