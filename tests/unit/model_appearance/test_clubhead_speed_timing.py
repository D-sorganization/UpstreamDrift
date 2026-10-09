"""GCV-20 (#11767): clubhead speed timing relative to impact.

The shared measure behind the acceptance: the peak clubhead (face-centre)
speed and the pre-contact speed at impact, each swing on its own clock with
impact at the sub-sample ball passage (OSV-10). The committed fixture
provenance must show the model peak within 5 ms of the capture peak, the
impact speed within 3 % of the capture's and the post-impact speed drop (the
ball's collision) within 10 % of the capture's, for the driver and the 7-iron.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.model_appearance import club_face as cf

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
PROVENANCE = ROOT / "tests/fixtures/club_face/provenance.json"
PEAK_TIMING_TOL_S = 0.005
IMPACT_SPEED_TOL = 0.03
POST_IMPACT_DROP_TOL = 0.10


def _swing(
    t: np.ndarray, *, collision: bool, brake_at_s: float | None = None
) -> np.ndarray:
    """Head on a 1 m circle: back to the top at t = 1, down to the ball at
    t = 1.4 still accelerating; with ``collision`` the ball takes 30 % of the
    speed and the head coasts on. ``brake_at_s`` instead holds the head at
    80 % of its angular rate from that time on (an early peak)."""
    theta = np.where(
        t < 1.0,
        np.pi / 2.0 * (1.0 - np.cos(np.pi * t)),
        np.pi * (1.0 - ((t - 1.0) / 0.4) ** 2),
    )
    rate = 2.0 * np.pi / 0.4  # |d theta / dt| at the ball
    if collision:
        theta = np.where(t > 1.4, -0.7 * rate * (t - 1.4), theta)
    if brake_at_s is not None:
        theta_b = np.pi * (1.0 - ((brake_at_s - 1.0) / 0.4) ** 2)
        rate_b = 2.0 * np.pi * (brake_at_s - 1.0) / 0.16
        theta = np.where(
            t > brake_at_s, theta_b - 0.8 * rate_b * (t - brake_at_s), theta
        )
    head = np.zeros((len(t), 3))
    head[:, 1] = np.sin(theta)
    head[:, 2] = 0.1 + (1.0 - np.cos(theta))
    return head


def test_speed_timing_peaks_at_the_ball_for_an_accelerating_strike() -> None:
    t = np.arange(0.0, 1.6, 1.0 / 360.0)
    timing = cf.clubhead_speed_timing(t, _swing(t, collision=True))
    assert timing.impact_time_s == pytest.approx(1.4, abs=1e-3)
    assert -1.0 / 360.0 <= timing.peak_minus_impact_s <= 0.0
    assert timing.impact_speed_mps == pytest.approx(2.0 * np.pi / 0.4, rel=0.01)
    assert timing.peak_speed_mps >= timing.impact_speed_mps


def test_impact_speed_uses_only_pre_contact_samples() -> None:
    """The segment straddling the ball mixes pre- and post-contact speed."""
    t = np.arange(0.0, 1.6, 1.0 / 360.0)
    with_ball = cf.clubhead_speed_timing(t, _swing(t, collision=True))
    without = cf.clubhead_speed_timing(t, _swing(t, collision=False))
    assert with_ball.impact_speed_mps == pytest.approx(without.impact_speed_mps)


def test_post_impact_speed_is_measured_after_the_contact() -> None:
    t = np.arange(0.0, 1.6, 1.0 / 360.0)
    timing = cf.clubhead_speed_timing(t, _swing(t, collision=True))
    assert timing.post_impact_speed_mps == pytest.approx(
        0.7 * 2.0 * np.pi / 0.4, rel=0.01
    )
    assert timing.post_impact_drop_mps == pytest.approx(
        0.3 * 2.0 * np.pi / 0.4, rel=0.03
    )


def test_peak_search_is_limited_to_the_impact_window() -> None:
    """A tracking glitch 150 ms after impact cannot be picked as the peak."""
    t = np.arange(0.0, 1.6, 1.0 / 360.0)
    head = _swing(t, collision=True)
    glitch = int(np.searchsorted(t, 1.55))
    head[glitch] += [0.0, 0.5, 0.0]
    timing = cf.clubhead_speed_timing(t, head)
    assert -1.0 / 360.0 <= timing.peak_minus_impact_s <= 0.0
    lo, hi = cf.SPEED_PEAK_WINDOW_S
    assert lo < 0.0 < hi


def test_speed_timing_detects_an_early_peak() -> None:
    """A head that brakes 25 ms before the ball reports an early peak."""
    t = np.arange(0.0, 1.6, 1.0 / 1000.0)
    timing = cf.clubhead_speed_timing(t, _swing(t, collision=False, brake_at_s=1.375))
    assert timing.peak_minus_impact_s < -0.020


def test_speed_timing_contracts() -> None:
    t = np.arange(0.0, 1.6, 1.0 / 360.0)
    head = _swing(t, collision=True)
    with pytest.raises(ValueError, match="clubhead"):
        cf.clubhead_speed_timing(t[:-1], head)
    bad = head.copy()
    bad[5, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        cf.clubhead_speed_timing(t, bad)
    with pytest.raises(ValueError, match="increase"):
        cf.clubhead_speed_timing(t[::-1], head)


def test_speed_timing_record_is_json_ready() -> None:
    t = np.arange(0.0, 1.6, 1.0 / 360.0)
    record = cf.clubhead_speed_timing(t, _swing(t, collision=True)).to_record()
    assert set(record) == {
        "impact_time_s",
        "peak_time_s",
        "peak_minus_impact_s",
        "peak_speed_mps",
        "impact_speed_mps",
        "post_impact_speed_mps",
        "post_impact_drop_mps",
    }
    json.dumps(record)


# ------------------------------------------- committed fixtures (GCV-20)
@pytest.mark.xfail(
    strict=True,
    reason=(
        "GCV-20 #11767 acceptance not met: the full re-solve at 5f8ee41fe5 "
        "meets the 5 ms peak timing (driver -3.7 ms, 7-iron -4.2 ms) but its "
        "impact speed is -7.4 % (driver) and -4.1 % (7-iron) against the 3 % "
        "limit, so the committed provenance fixture is not regenerated "
        "(DESIGN_DECISIONS decision 11 amendment) and carries no "
        "after.speed_timing record yet"
    ),
)
@pytest.mark.parametrize("club", ["driver", "iron7"])
def test_committed_fixture_speed_timing_matches_the_capture(club: str) -> None:
    """Regenerated fixtures: peak within 5 ms, impact speed within 3 %, and
    the ball's post-impact speed drop within 10 %."""
    record = json.loads(PROVENANCE.read_text(encoding="utf-8"))["clubs"][club]
    speed = record["after"]["speed_timing"]
    model, capture = speed["model"], speed["capture"]
    lead = model["peak_minus_impact_s"] - capture["peak_minus_impact_s"]
    assert abs(lead) <= PEAK_TIMING_TOL_S, (club, model, capture)
    ratio = model["impact_speed_mps"] / capture["impact_speed_mps"]
    assert abs(ratio - 1.0) <= IMPACT_SPEED_TOL, (club, model, capture)
    drop = model["post_impact_drop_mps"] / capture["post_impact_drop_mps"]
    assert abs(drop - 1.0) <= POST_IMPACT_DROP_TOL, (club, model, capture)
