"""Shooting windows use observed clocks without inventing boundary samples."""

import numpy as np
import pytest

from src.shared.python.motion_matching.shooting_schedule import sampled_shooting_windows

pytestmark = pytest.mark.unit


def test_schedule_uses_actual_samples_and_detaches_windows() -> None:
    time = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 0.85, 1.0])
    windows = sampled_shooting_windows(time, [0.4, 0.8, 0.85])
    assert len(windows) == 3
    np.testing.assert_array_equal(windows[0], [0.0, 0.2, 0.4])
    np.testing.assert_array_equal(windows[1], [0.4, 0.6, 0.8])
    np.testing.assert_array_equal(windows[2], [0.8, 0.85])
    time[:] = 2.0
    assert windows[0][0] == 0.0
    assert not windows[0].flags.writeable


@pytest.mark.parametrize(
    "nodes", [[], [0.0], [0.4, 0.2], [0.2, 0.2], [0.3], [1.1], [float("nan")]]
)
def test_schedule_rejects_missing_or_invalid_nodes(nodes: list[float]) -> None:
    with pytest.raises(ValueError):
        sampled_shooting_windows(np.array([0.0, 0.2, 0.4, 1.0]), nodes)


@pytest.mark.parametrize(
    "time", [[0.1, 0.4], [0.0, 0.4, 0.2], [0.0, 0.2, 0.2], [0.0, float("nan")]]
)
def test_schedule_rejects_invalid_capture_clock(time: list[float]) -> None:
    with pytest.raises(ValueError):
        sampled_shooting_windows(np.array(time), [0.4])
