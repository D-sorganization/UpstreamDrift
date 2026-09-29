"""Tests for the canonical swing event detector module (Issue #11014).

Validates:
- SwingEventFrames dataclass postconditions and immutability.
- detect_swing_events preconditions, postconditions, and finish_fraction parameter.
- peak_speed_index helper and preconditions.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from src.shared.python.analysis.swing_events import (
    SwingEventFrames,
    detect_swing_events,
    peak_speed_index,
)

pytestmark = pytest.mark.unit


# ==================== SwingEventFrames Tests ====================


def test_swing_event_frames_valid() -> None:
    """Valid monotonic frame ordering creates a valid instance."""
    frames = SwingEventFrames(address=0, top=10, peak=20, finish=30)
    assert frames.address == 0
    assert frames.top == 10
    assert frames.peak == 20
    assert frames.finish == 30


def test_swing_event_frames_all_equal() -> None:
    """Non-decreasing ordering allows adjacent equal frame indices."""
    frames = SwingEventFrames(address=5, top=5, peak=5, finish=5)
    assert frames.address == 5
    assert frames.top == 5
    assert frames.peak == 5
    assert frames.finish == 5


@pytest.mark.parametrize(
    ("address", "top", "peak", "finish"),
    [
        (-1, 10, 20, 30),  # address < 0
        (15, 10, 20, 30),  # top < address
        (10, 25, 20, 30),  # peak < top
        (10, 20, 35, 30),  # finish < peak
    ],
)
def test_swing_event_frames_postconditions_raise(
    address: int, top: int, peak: int, finish: int
) -> None:
    """Postcondition raises ValueError if frame indices are not 0 <= address <= top <= peak <= finish."""
    with pytest.raises(ValueError):
        SwingEventFrames(address=address, top=top, peak=peak, finish=finish)


def test_swing_event_frames_frozen() -> None:
    """SwingEventFrames is frozen and attributes cannot be mutated."""
    frames = SwingEventFrames(address=0, top=10, peak=20, finish=30)
    with pytest.raises(FrozenInstanceError):
        frames.address = 1  # type: ignore[misc]


# ==================== peak_speed_index Tests ====================


def test_peak_speed_index_valid() -> None:
    """peak_speed_index returns index of global maximum."""
    speed = [1.0, 5.0, 40.0, 20.0, 2.0]
    assert peak_speed_index(speed) == 2


def test_peak_speed_index_numpy() -> None:
    """peak_speed_index works with numpy arrays."""
    arr = np.array([0.5, 2.0, 8.0, 1.0])
    assert peak_speed_index(arr) == 2


@pytest.mark.parametrize(
    "invalid_input",
    [
        [],  # empty
        [[1.0, 2.0], [3.0, 4.0]],  # 2-D
        [1.0, np.nan, 2.0],  # NaN
        [1.0, np.inf, 2.0],  # Inf
    ],
)
def test_peak_speed_index_preconditions(invalid_input: object) -> None:
    """peak_speed_index raises ValueError on non-1D, empty, or non-finite inputs."""
    with pytest.raises(ValueError):
        peak_speed_index(invalid_input)  # type: ignore[arg-type]


# ==================== detect_swing_events Tests ====================


def _make_dummy_speed(n: int = 100) -> np.ndarray:
    """Create a simple valid 1-D speed profile."""
    t = np.linspace(0, 1, n)
    return np.sin(np.pi * t) * 10.0


@pytest.mark.parametrize(
    "invalid_fps",
    [0.0, -10.0, float("nan"), float("inf")],
)
def test_detect_swing_events_fps_preconditions(invalid_fps: float) -> None:
    """fps must be positive and finite."""
    speed = _make_dummy_speed()
    with pytest.raises(ValueError):
        detect_swing_events(speed, invalid_fps)


@pytest.mark.parametrize(
    "invalid_speed",
    [
        [],  # empty (< 3 samples)
        [1.0, 2.0],  # 2 samples (< 3 samples)
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],  # 2-D
        [1.0, np.nan, 3.0],  # NaN
        [1.0, float("inf"), 3.0],  # Inf
    ],
)
def test_detect_swing_events_speed_preconditions(invalid_speed: object) -> None:
    """speed must be 1-D, finite, and have >= 3 samples."""
    with pytest.raises(ValueError):
        detect_swing_events(invalid_speed, 100.0)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "invalid_quiet_fraction",
    [0.0, 1.0, -0.1, 1.5, float("nan")],
)
def test_detect_swing_events_quiet_fraction_preconditions(
    invalid_quiet_fraction: float,
) -> None:
    """quiet_fraction must be in (0, 1) and finite."""
    speed = _make_dummy_speed()
    with pytest.raises(ValueError):
        detect_swing_events(speed, 100.0, quiet_fraction=invalid_quiet_fraction)


@pytest.mark.parametrize("invalid_quiet_s", [0.0, -0.1, float("nan"), float("inf")])
def test_detect_swing_events_quiet_s_preconditions(invalid_quiet_s: float) -> None:
    """quiet_s must be positive and finite."""
    speed = _make_dummy_speed()
    with pytest.raises(ValueError):
        detect_swing_events(speed, 100.0, quiet_s=invalid_quiet_s)


@pytest.mark.parametrize("invalid_downswing_s", [0.0, -0.5, float("nan"), float("inf")])
def test_detect_swing_events_max_downswing_s_preconditions(
    invalid_downswing_s: float,
) -> None:
    """max_downswing_s must be positive and finite."""
    speed = _make_dummy_speed()
    with pytest.raises(ValueError):
        detect_swing_events(speed, 100.0, max_downswing_s=invalid_downswing_s)


@pytest.mark.parametrize("invalid_finish_fraction", [0.0, 1.0, -0.2, 1.2, float("nan")])
def test_detect_swing_events_finish_fraction_preconditions(
    invalid_finish_fraction: float,
) -> None:
    """finish_fraction must be in (0, 1) and finite when provided."""
    speed = _make_dummy_speed()
    with pytest.raises(ValueError):
        detect_swing_events(speed, 100.0, finish_fraction=invalid_finish_fraction)


def test_detect_swing_events_finish_fraction_effect() -> None:
    """Passing a higher finish_fraction detects finish earlier (closer to peak)."""
    n = 200
    fps = 100.0
    t = np.arange(n, dtype=float)
    # Peak at 100, then gradual decay
    speed = 0.1 + 30.0 * np.exp(-(((t - 100.0) / 30.0) ** 2))

    events_default = detect_swing_events(speed, fps)  # finish_fraction=None -> 0.05
    events_higher = detect_swing_events(speed, fps, finish_fraction=0.3)

    # 30% threshold is reached earlier than 5% threshold
    assert events_higher.finish < events_default.finish
    assert events_higher.peak == events_default.peak
    assert events_higher.top == events_default.top
    assert events_higher.address == events_default.address
