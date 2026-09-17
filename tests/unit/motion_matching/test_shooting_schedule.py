"""Shooting windows preserve observed sample times and boundary ownership."""

import numpy as np
import pytest
from src.shared.python.motion_matching.shooting_schedule import sampled_shooting_windows

pytestmark = pytest.mark.unit


def test_windows_preserve_samples_and_shared_endpoint() -> None:
    clock = np.arange(361) / 360
    windows = sampled_shooting_windows(clock, (0.6, 0.85))
    np.testing.assert_array_equal(windows[0], clock[:217])
    np.testing.assert_array_equal(windows[1], clock[216:307])
    assert not windows[0].flags.writeable
    clock[:] = 9
    assert windows[0][0] == 0


@pytest.mark.parametrize("nodes", [(0.6001,), (0.6, 0.5), (float("nan"),)])
def test_reject_off_grid_or_invalid_nodes(nodes: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        sampled_shooting_windows(np.arange(361) / 360, nodes)
