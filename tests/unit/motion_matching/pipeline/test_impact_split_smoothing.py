"""GCV-20 (#11767): the reference low-pass must not filter across impact.

The ball collision removes roughly a quarter of the clubhead speed within one
capture sample. A zero-phase low-pass run across that step spreads the drop
over its whole kernel, so the filtered head starts slowing about 25 ms before
the ball. ``smooth_reference(..., impact_index=k)`` filters the pre-contact
samples ``[0, k]`` and the post-contact samples ``[k + 1, n)`` separately.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import butter, filtfilt

from src.shared.python.motion_matching.pipeline.constants import (
    RATE_HZ,
    REFERENCE_CUTOFF_HZ,
)
from src.shared.python.motion_matching.pipeline.reference import smooth_reference

pytestmark = pytest.mark.unit


def _swing_with_collision(n: int = 500, k: int = 360) -> tuple[np.ndarray, int]:
    """One coordinate accelerating smoothly up to sample ``k``, then a 30 %
    velocity step down (the ball), then a slow coast."""
    dt = 1.0 / RATE_HZ
    t = np.arange(n) * dt
    t_imp = t[k]
    vel = np.where(t <= t_imp, 40.0 * (t / t_imp) ** 3, 0.7 * 40.0)
    q = np.cumsum(vel) * dt
    return q[:, None], k


def test_unsplit_filter_moves_the_peak_before_the_collision() -> None:
    """The defect, reproduced: an unsplit filter peaks well before impact."""
    q, k = _swing_with_collision()
    v = np.diff(smooth_reference(q, RATE_HZ, REFERENCE_CUTOFF_HZ)[:, 0]) * RATE_HZ
    assert k - int(np.argmax(v)) >= 5  # >= 14 ms early at 360 Hz


def test_split_filter_keeps_the_peak_at_the_last_pre_contact_sample() -> None:
    q, k = _swing_with_collision()
    raw_v = np.diff(q[:, 0]) * RATE_HZ
    smooth = smooth_reference(q, RATE_HZ, REFERENCE_CUTOFF_HZ, impact_index=k)
    v = np.diff(smooth[:, 0]) * RATE_HZ
    peak = int(np.argmax(v))
    assert k - 2 <= peak <= k - 1  # the last whole pre-contact segments
    assert v[k - 1] == pytest.approx(raw_v[k - 1], rel=0.03)


def test_split_filter_equals_two_independent_filters() -> None:
    rng = np.random.default_rng(11767)
    q = np.cumsum(rng.normal(size=(300, 3)), axis=0)
    out = smooth_reference(q, RATE_HZ, REFERENCE_CUTOFF_HZ, impact_index=200)
    b, a = butter(4, REFERENCE_CUTOFF_HZ / (0.5 * RATE_HZ))
    pre = filtfilt(b, a, q[:201], axis=0, padlen=60)
    post = filtfilt(b, a, q[201:], axis=0, padlen=60)
    np.testing.assert_array_equal(out, np.vstack([pre, post]))


def test_no_impact_index_keeps_the_legacy_filter_bit_for_bit() -> None:
    rng = np.random.default_rng(7)
    q = rng.normal(size=(120, 2))
    np.testing.assert_array_equal(
        smooth_reference(q, RATE_HZ, 12.0, impact_index=None),
        smooth_reference(q, RATE_HZ, 12.0),
    )


@pytest.mark.parametrize("bad", [0, -1, 119, 120, 500])
def test_impact_index_must_leave_samples_on_both_sides(bad: int) -> None:
    with pytest.raises(ValueError, match="impact_index"):
        smooth_reference(np.zeros((120, 2)), RATE_HZ, 12.0, impact_index=bad)


@pytest.mark.parametrize("bad", [True, 3.0, "40"])
def test_impact_index_must_be_an_integer(bad: object) -> None:
    with pytest.raises(TypeError, match="impact_index"):
        smooth_reference(np.zeros((120, 2)), RATE_HZ, 12.0, impact_index=bad)  # type: ignore[arg-type]


def test_numpy_integer_impact_index_is_accepted() -> None:
    out = smooth_reference(np.zeros((120, 2)), RATE_HZ, 12.0, impact_index=np.int64(60))
    assert out.shape == (120, 2)


# ------------------------------------------- impact frame from the capture
def _capture_centres(n: int = 600, rate: float = RATE_HZ) -> np.ndarray:
    """Face centre on a 1 m circle reaching the ball (address) at t = 1.4 s."""
    t = np.arange(n) / rate
    theta = np.where(
        t < 1.0,
        np.pi / 2.0 * (1.0 - np.cos(np.pi * t)),
        np.pi * (1.0 - ((t - 1.0) / 0.4) ** 2),
    )
    head = np.zeros((n, 3))
    head[:, 1] = np.sin(theta)
    head[:, 2] = 0.1 + (1.0 - np.cos(theta))
    return head


def test_capture_impact_index_is_the_last_pre_contact_capture_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.shared.python.motion_matching import club_face_target as cft

    rate = 333.0  # the ball passage falls between two samples
    centres = _capture_centres(rate=rate)
    centres[100:104] = np.nan  # a triad gap is interpolated, not fatal
    monkeypatch.setattr(
        cft, "observe_capture_face", lambda *a, **k: (np.zeros_like(centres), centres)
    )
    times = np.arange(len(centres)) / rate
    k = cft.capture_impact_index(times, None, None, (), {}, {})
    assert times[k] <= 1.4 < times[k + 1]


def test_capture_impact_index_rejects_mismatched_times(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.shared.python.motion_matching import club_face_target as cft

    centres = _capture_centres()
    monkeypatch.setattr(
        cft, "observe_capture_face", lambda *a, **k: (np.zeros_like(centres), centres)
    )
    with pytest.raises(ValueError, match="times"):
        cft.capture_impact_index(np.arange(10.0), None, None, (), {}, {})
