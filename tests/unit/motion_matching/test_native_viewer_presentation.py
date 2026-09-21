"""Unit tests for presentation frame pacing and Gepetto locking (MV-02, #10478)."""

from __future__ import annotations

import numpy as np
import pytest

from src.engines.physics_engines.pinocchio.python.viewer_presentation import (
    gepetto_playback_lock,
    presentation_frames,
)

pytestmark = [pytest.mark.unit]


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def read(self) -> float:
        return self.now

    def sleep(self, duration: float) -> None:
        self.now += duration


def test_slow_renderer_skips_stale_frames_and_reaches_endpoint() -> None:
    clock = FakeClock()
    frames: list[int] = []
    times = np.linspace(0, 1, 361)
    for index in presentation_frames(times, 30.0, 1.0, clock.read, clock.sleep):
        frames.append(index)
        clock.now += 0.1  # Deliberately slower than the presentation budget.
    assert frames[0] == 0 and frames[-1] == 360
    assert len(frames) < 12
    assert clock.now < 1.3
    assert frames == sorted(set(frames))


def test_quarter_speed_preserves_capture_timestamps_and_limits_updates() -> None:
    clock = FakeClock()
    times = np.array([2.0, 2.2, 2.6, 3.0])
    original = times.copy()
    frames = list(presentation_frames(times, 30.0, 0.25, clock.read, clock.sleep))
    np.testing.assert_array_equal(times, original)
    assert frames == [0, 1, 2, 3]
    assert 4.0 <= clock.now < 4.04


@pytest.mark.parametrize(
    "fps,speed",
    [
        (0.0, 1.0),
        (30.0, 0.0),
        (float("nan"), 1.0),
        (30.0, float("inf")),
    ],
)
def test_invalid_presentation_options_fail(fps: float, speed: float) -> None:
    with pytest.raises(ValueError, match="Presentation fps and speed"):
        list(presentation_frames(np.array([0.0, 1.0]), fps, speed))


def test_gepetto_playback_lock_acquires_and_releases() -> None:
    with gepetto_playback_lock():
        pass  # Lock should acquire and release cleanly on any platform
