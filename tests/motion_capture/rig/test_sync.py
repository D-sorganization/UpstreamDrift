"""Strobe alignment across cameras with known synthetic offsets."""

from __future__ import annotations

import numpy as np
import pytest

from src.motion_capture.rig.sources import HOST_MONOTONIC
from src.motion_capture.rig.sync import (
    FLASH_METHOD,
    TimingRecord,
    align_views,
    detect_flash_index,
    flash_event,
    median_interval_ns,
    rate_deviation_ppm,
)

pytestmark = pytest.mark.unit

PERIOD_60 = 16_666_667  # ns, exact quotient rounded
SYNTH_PERIOD_60 = int(1e9 / 60)  # SyntheticFrameSource truncates


def _series(
    n: int, *, t0_ns: int, period_ns: int, flash_at: int | None
) -> tuple[np.ndarray, np.ndarray]:
    t = t0_ns + period_ns * np.arange(n, dtype=np.int64)
    b = np.full(n, 32.0)
    if flash_at is not None:
        b[flash_at:] = 240.0  # strobe stays on for the rest of the clip
    return t, b


def test_detect_flash_index_finds_first_bright_frame_after_baseline() -> None:
    b = np.array([30, 31, 30, 32, 31, 200, 210], dtype=float)
    assert detect_flash_index(b) == 5
    assert detect_flash_index(b, min_jump=500) is None
    with pytest.raises(Exception, match="min_jump"):
        detect_flash_index(b, min_jump=0)


def test_flash_event_uncertainty_is_the_preceding_frame_interval() -> None:
    t, b = _series(20, t0_ns=1_000, period_ns=PERIOD_60, flash_at=8)
    event = flash_event(t, b)
    assert event is not None
    assert (event.index, event.t_ns, event.uncertainty_ns) == (
        8,
        1_000 + 8 * PERIOD_60,
        PERIOD_60,
    )
    assert flash_event(t, np.full(20, 32.0)) is None
    with pytest.raises(Exception, match="align"):
        flash_event(t, b[:-1])


def test_interval_and_rate_deviation() -> None:
    t, _ = _series(10, t0_ns=0, period_ns=25_000_000, flash_at=None)  # 40 fps
    assert median_interval_ns(t) == 25_000_000
    assert rate_deviation_ppm(t, 60) == pytest.approx(500_000, rel=1e-6)  # 50 % slow
    assert rate_deviation_ppm(t, 40) == pytest.approx(0.0, abs=1e-6)
    assert median_interval_ns(np.array([5])) is None


def test_align_views_recovers_known_offsets_with_quadrature_uncertainty() -> None:
    period_b = 20_000_000  # 50 fps camera
    series = {
        "ref": _series(30, t0_ns=0, period_ns=PERIOD_60, flash_at=10),
        "b": _series(30, t0_ns=7_000_000, period_ns=period_b, flash_at=9),
        "c": _series(30, t0_ns=-3_000_000, period_ns=PERIOD_60, flash_at=11),
    }
    record = align_views("ref", series, {"ref": 60, "b": 50, "c": 60})
    assert isinstance(record, TimingRecord)
    assert record.status == "available"
    assert record.method == FLASH_METHOD and record.clock_domain == HOST_MONOTONIC
    by_view = {v.view: v for v in record.views}
    assert by_view["ref"].offset_ns == 0
    assert by_view["b"].offset_ns == 7_000_000 + 9 * period_b - 10 * PERIOD_60
    assert by_view["c"].offset_ns == -3_000_000 + 11 * PERIOD_60 - 10 * PERIOD_60
    assert by_view["b"].uncertainty_ns == round((period_b**2 + PERIOD_60**2) ** 0.5)
    assert by_view["ref"].uncertainty_ns == round((2 * PERIOD_60**2) ** 0.5)
    assert by_view["b"].rate_deviation_ppm == pytest.approx(0.0, abs=1e-6)
    assert by_view["ref"].rate_deviation_ppm == pytest.approx(0.0, abs=1.0)


def test_align_views_is_unavailable_when_a_view_lacks_the_strobe() -> None:
    series = {
        "ref": _series(20, t0_ns=0, period_ns=PERIOD_60, flash_at=5),
        "dark": _series(20, t0_ns=0, period_ns=PERIOD_60, flash_at=None),
    }
    record = align_views("ref", series, {"ref": 60, "dark": 60})
    assert record.status == "unavailable"
    dark = next(v for v in record.views if v.view == "dark")
    assert dark.status == "unavailable" and dark.reason == "no strobe detected"
    assert dark.offset_ns is None
    ref = next(v for v in record.views if v.view == "ref")
    assert ref.status == "available" and ref.offset_ns == 0


def test_align_views_is_unavailable_when_the_reference_lacks_the_strobe() -> None:
    series = {
        "ref": _series(20, t0_ns=0, period_ns=PERIOD_60, flash_at=None),
        "b": _series(20, t0_ns=0, period_ns=PERIOD_60, flash_at=5),
    }
    record = align_views("ref", series, {"ref": 60, "b": 60})
    assert record.status == "unavailable"
    b = next(v for v in record.views if v.view == "b")
    assert b.reason == "reference view has no strobe" and b.event_index == 5
    with pytest.raises(Exception, match="reference_view"):
        align_views("missing", series, {"ref": 60, "b": 60})


def test_session_collects_timing_when_asked() -> None:
    from src.motion_capture.rig.plan import CameraBinding, RigPlan
    from src.motion_capture.rig.session import CaptureSession, CaptureTuning
    from src.motion_capture.rig.sources import SyntheticFrameSource

    plan = RigPlan(
        name="strobe",
        cameras=(
            CameraBinding(view="ref", serial="1"),
            CameraBinding(view="side", serial="2"),
        ),
    )
    sources = {
        "ref": SyntheticFrameSource("1", flash_at=12),
        "side": SyntheticFrameSource("2", flash_at=12),
    }
    manifest = CaptureSession(
        plan,
        sources,
        duration_s=5,
        tuning=CaptureTuning(max_frames=40, collect_timing=True),
    ).run()
    timing = manifest.timing
    assert timing["status"] == "available"
    assert timing["reference_view"] == "ref" and timing["method"] == FLASH_METHOD
    side = next(v for v in timing["views"] if v["view"] == "side")
    assert side["event_index"] == 12
    assert side["uncertainty_ns"] == round((2 * SYNTH_PERIOD_60**2) ** 0.5)
    # Sources open a few microseconds apart; the offset is that gap, not a frame.
    assert abs(side["offset_ns"]) < SYNTH_PERIOD_60


def test_session_timing_is_unavailable_without_a_strobe() -> None:
    from src.motion_capture.rig.plan import CameraBinding, RigPlan
    from src.motion_capture.rig.session import CaptureSession, CaptureTuning
    from src.motion_capture.rig.sources import SyntheticFrameSource

    plan = RigPlan(name="dark", cameras=(CameraBinding(view="ref", serial="1"),))
    manifest = CaptureSession(
        plan,
        {"ref": SyntheticFrameSource("1")},
        duration_s=5,
        tuning=CaptureTuning(max_frames=20, collect_timing=True),
    ).run()
    assert manifest.timing["status"] == "unavailable"
    assert manifest.timing["views"][0]["reason"] == "no strobe detected"
    plain = CaptureSession(
        plan,
        {"ref": SyntheticFrameSource("1")},
        duration_s=5,
        tuning=CaptureTuning(max_frames=20),
    ).run()
    assert plain.timing == {}
