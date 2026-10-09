"""GCV-20 (#11767): the pre-contact low-pass must keep the release.

A fixed 12 Hz cutoff removes the driver's late release: the filtered reference
peaks 13 ms before impact and 10 % slow, while the unfiltered reference peaks
with the capture. ``release_preserving_cutoff`` picks the lowest candidate
cutoff whose pre-contact filtered clubhead keeps the unfiltered reference's
speed-peak timing (within one sample) and impact speed (within 1 %);
``smooth_reference(..., pre_contact_cutoff_hz=f)`` applies it to the
pre-contact samples only, the post-contact samples keep the base cutoff.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.signal import butter, filtfilt

from src.shared.python.motion_matching.pipeline.constants import (
    RATE_HZ,
    RELEASE_CUTOFF_CANDIDATES_HZ,
    TRACKING_CUTOFF_HZ,
)
from src.shared.python.motion_matching.pipeline.reference import (
    smooth_lane,
    smooth_reference,
)
from src.shared.python.motion_matching.pipeline.release_cutoff import (
    ReleaseCutoff,
    release_preserving_cutoff,
)

pytestmark = pytest.mark.unit

RADIUS_M = 1.2
IMPACT_S = 1.4


def _swing(
    ramp_power: float, burst_s: float = 0.0
) -> tuple[np.ndarray, np.ndarray, int]:
    """One club angle: back to the top at 1 s, down through the ball at 1.4 s.

    The downswing angular speed rises as ``x**ramp_power`` (``x`` the
    downswing fraction), optionally with a release burst of width ``burst_s``
    just before impact; the ball removes a quarter of the speed. Returns
    ``(time, q (n, 1), last pre-contact index)``.
    """
    t = np.arange(int(1.8 * RATE_HZ)) / RATE_HZ
    w = np.zeros_like(t)
    back = t <= 1.0
    w[back] = -1.5 * np.pi * np.sin(np.pi * t[back])
    down = (t > 1.0) & (t <= IMPACT_S)
    x = (t[down] - 1.0) / (IMPACT_S - 1.0)
    g = x**ramp_power
    if burst_s:
        g = g + 2.0 * np.exp(-(((t[down] - (IMPACT_S - burst_s)) / burst_s) ** 2))
    w[down] = g * 3.0 * RATE_HZ / g.sum()
    after = t > IMPACT_S
    w[after] = 0.75 * w[down][-1] * np.exp(-(t[after] - IMPACT_S) / 0.15)
    phi = np.cumsum(w) / RATE_HZ
    return t, (phi - phi[0])[:, None], int(np.flatnonzero(down)[-1])


def _head(q: np.ndarray) -> np.ndarray:
    phi = q[:, 0]
    return np.column_stack(
        [np.zeros_like(phi), RADIUS_M * np.sin(phi), 0.1 + RADIUS_M * (1 - np.cos(phi))]
    )


# ------------------------------------------------------- pre-contact cutoff
def test_pre_contact_cutoff_filters_each_side_at_its_own_cutoff() -> None:
    rng = np.random.default_rng(11767)
    q = np.cumsum(rng.normal(size=(300, 3)), axis=0)
    out = smooth_reference(
        q, RATE_HZ, TRACKING_CUTOFF_HZ, impact_index=200, pre_contact_cutoff_hz=25.0
    )
    nyquist = 0.5 * RATE_HZ
    b_pre, a_pre = butter(4, 25.0 / nyquist)
    b_post, a_post = butter(4, TRACKING_CUTOFF_HZ / nyquist)
    pre = filtfilt(b_pre, a_pre, q[:201], axis=0, padlen=60)
    post = filtfilt(b_post, a_post, q[201:], axis=0, padlen=60)
    np.testing.assert_array_equal(out, np.vstack([pre, post]))


def test_pre_contact_cutoff_equal_to_the_base_keeps_the_split_filter() -> None:
    rng = np.random.default_rng(3)
    q = rng.normal(size=(200, 2))
    np.testing.assert_array_equal(
        smooth_reference(
            q, RATE_HZ, 12.0, impact_index=120, pre_contact_cutoff_hz=12.0
        ),
        smooth_reference(q, RATE_HZ, 12.0, impact_index=120),
    )


def test_pre_contact_cutoff_needs_the_impact_split() -> None:
    with pytest.raises(ValueError, match="impact_index"):
        smooth_reference(np.zeros((120, 2)), RATE_HZ, 12.0, pre_contact_cutoff_hz=25.0)


@pytest.mark.parametrize("bad", [0.0, -5.0, 180.0, float("nan")])
def test_pre_contact_cutoff_must_lie_inside_the_band(bad: float) -> None:
    with pytest.raises(ValueError, match="pre_contact_cutoff_hz"):
        smooth_reference(
            np.zeros((120, 2)),
            RATE_HZ,
            12.0,
            impact_index=60,
            pre_contact_cutoff_hz=bad,
        )


def test_smooth_lane_applies_the_lane_release_cutoff() -> None:
    rng = np.random.default_rng(5)
    q = rng.normal(size=(200, 2))
    lane = SimpleNamespace(
        rate_hz=RATE_HZ, impact_index=120, pre_contact_cutoff_hz=25.0
    )
    np.testing.assert_array_equal(
        smooth_lane(q, lane, 12.0),
        smooth_reference(
            q, RATE_HZ, 12.0, impact_index=120, pre_contact_cutoff_hz=25.0
        ),
    )
    legacy = SimpleNamespace(rate_hz=RATE_HZ, impact_index=120)
    np.testing.assert_array_equal(
        smooth_lane(q, legacy, 12.0),
        smooth_reference(q, RATE_HZ, 12.0, impact_index=120),
    )


# ---------------------------------------------------- release preservation
def test_a_release_inside_the_base_band_keeps_the_base_cutoff() -> None:
    t, q, k = _swing(ramp_power=0.5)
    result = release_preserving_cutoff(t, q, k, _head, base_cutoff_hz=12.0)
    assert result.cutoff_hz == 12.0
    assert result.preserved


def test_a_sharp_release_raises_the_pre_contact_cutoff() -> None:
    t, q, k = _swing(ramp_power=1.0)
    result = release_preserving_cutoff(t, q, k, _head, base_cutoff_hz=12.0)
    assert result.preserved
    assert result.cutoff_hz > 12.0
    # Every lower candidate lost the release; the chosen one keeps it.
    rows = {row["cutoff_hz"]: row for row in result.candidates}
    assert not any(rows[f]["preserved"] for f in rows if f < result.cutoff_hz)
    chosen = rows[result.cutoff_hz]
    raw = result.unfiltered
    assert (
        abs(
            chosen["last_pre_contact_speed_mps"] / raw["last_pre_contact_speed_mps"] - 1
        )
        <= 0.01
    )
    assert (
        abs(chosen["peak_minus_split_s"] - raw["peak_minus_split_s"])
        <= 1.0 / RATE_HZ + 1e-12
    )


def test_a_release_no_candidate_keeps_is_flagged_not_hidden() -> None:
    t, q, k = _swing(ramp_power=2.0, burst_s=0.006)
    result = release_preserving_cutoff(
        t, q, k, _head, base_cutoff_hz=12.0, candidates_hz=(12.0, 15.0, 18.0)
    )
    assert not result.preserved
    assert result.cutoff_hz == 18.0  # the widest candidate, flagged in the receipt
    assert result.to_record()["preserved"] is False


def test_the_record_lists_every_candidate() -> None:
    t, q, k = _swing(ramp_power=3.0)
    record = release_preserving_cutoff(t, q, k, _head, base_cutoff_hz=12.0).to_record()
    assert [row["cutoff_hz"] for row in record["candidates"]] == list(
        RELEASE_CUTOFF_CANDIDATES_HZ
    )
    assert set(record) >= {
        "cutoff_hz",
        "base_cutoff_hz",
        "preserved",
        "unfiltered",
        "timing_tol_s",
        "speed_tol",
        "candidates",
    }


def test_the_default_candidates_start_at_the_tracking_cutoff() -> None:
    assert RELEASE_CUTOFF_CANDIDATES_HZ[0] == TRACKING_CUTOFF_HZ
    assert list(RELEASE_CUTOFF_CANDIDATES_HZ) == sorted(
        set(RELEASE_CUTOFF_CANDIDATES_HZ)
    )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"candidates_hz": ()}, "candidates_hz"),
        ({"candidates_hz": (15.0, 12.0)}, "candidates_hz"),
        ({"candidates_hz": (12.0, 200.0)}, "candidates_hz"),
        ({"base_cutoff_hz": 10.0}, "base_cutoff_hz"),
        ({"speed_tol": 0.0}, "speed_tol"),
    ],
)
def test_release_cutoff_contracts(kwargs: dict, match: str) -> None:
    t, q, k = _swing(ramp_power=1.0)
    args = {"base_cutoff_hz": 12.0, **kwargs}
    with pytest.raises(ValueError, match=match):
        release_preserving_cutoff(t, q, k, _head, **args)


def test_release_cutoff_rejects_mismatched_time() -> None:
    t, q, k = _swing(ramp_power=1.0)
    with pytest.raises(ValueError, match="time"):
        release_preserving_cutoff(t[:-1], q, k, _head, base_cutoff_hz=12.0)


def test_release_cutoff_result_is_frozen() -> None:
    t, q, k = _swing(ramp_power=1.0)
    result = release_preserving_cutoff(t, q, k, _head, base_cutoff_hz=12.0)
    assert isinstance(result, ReleaseCutoff)
    with pytest.raises(AttributeError):
        result.cutoff_hz = 40.0  # type: ignore[misc]


# ------------------------------------------------------ lane and receipt
def _lane(impact_index: int | None, times: np.ndarray):
    from src.shared.python.motion_matching.pipeline.lane import Lane

    lane = object.__new__(Lane)  # no capture needed for the cutoff selection
    lane.times = times
    lane.impact_index = impact_index
    lane.impact_time_s = None
    lane.impact_split_reason = "test"
    lane.pre_contact_cutoff_hz = None
    lane.release_cutoff = None
    return lane


def test_lane_selects_and_reports_the_release_cutoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.shared.python.motion_matching import club_face_target as cft

    monkeypatch.setattr(cft, "model_face_centres", lambda kin, q, spec: _head(q))
    t, q, k = _swing(ramp_power=1.0)
    lane = _lane(k, t)
    lane.select_release_cutoff(None, {}, q)
    assert lane.pre_contact_cutoff_hz is not None and lane.pre_contact_cutoff_hz > 12.0
    report = lane.impact_split_report()
    assert report["pre_contact_cutoff_hz"] == lane.pre_contact_cutoff_hz
    assert report["release_cutoff"]["preserved"] is True


def test_lane_without_an_impact_split_keeps_the_base_cutoff() -> None:
    t, q, _ = _swing(ramp_power=3.0)
    lane = _lane(None, t)
    lane.select_release_cutoff(None, {}, q)
    assert lane.pre_contact_cutoff_hz is None
    assert "pre_contact_cutoff_hz" not in lane.impact_split_report()
    assert lane.impact_split_report()["release_cutoff"]["source"] == "no impact split"


def test_lane_keeps_the_base_cutoff_when_the_clubhead_path_is_not_finite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.shared.python.motion_matching import club_face_target as cft

    monkeypatch.setattr(
        cft, "model_face_centres", lambda kin, q, spec: _head(q) * np.nan
    )
    t, q, k = _swing(ramp_power=3.0)
    lane = _lane(k, t)
    lane.select_release_cutoff(None, {}, q)
    assert lane.pre_contact_cutoff_hz is None
    assert lane.release_cutoff["source"].startswith("unavailable")


def test_the_segment_straddling_the_split_is_not_measured() -> None:
    """The segment joining the two filtered halves never decides the choice
    (on the 7-iron it read 41.1 m/s where the release had dropped to 36.4)."""
    t, q, k = _swing(ramp_power=3.0)

    def jumpy(rows: np.ndarray) -> np.ndarray:
        head = _head(rows)
        head[k + 1 :] += [0.0, 0.5, 0.0]  # a huge post-split jump
        return head

    assert release_preserving_cutoff(
        t, q, k, jumpy, base_cutoff_hz=12.0
    ) == release_preserving_cutoff(t, q, k, _head, base_cutoff_hz=12.0)


def test_model_face_centres_places_the_face_centre_with_the_frame_pose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.shared.python.motion_matching import club_face_target as cft

    monkeypatch.setattr(
        cft, "face_centre_in_frame", lambda spec, frame: np.array([1.0, 0, 0])
    )
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    class Kin:
        def body_poses(self, row, frames):
            return {frames[0]: (rot, np.array([row[0], 0.0, 0.0]))}

    out = cft.model_face_centres(Kin(), np.array([[0.0], [2.0]]), {})
    np.testing.assert_allclose(out, [[0.0, 1.0, 0.0], [2.0, 1.0, 0.0]])
    with pytest.raises(ValueError, match="frames"):
        cft.model_face_centres(Kin(), np.zeros(3), {})
