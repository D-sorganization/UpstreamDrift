"""Exact sampled windows for configurable shooting horizon experiments."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def sampled_shooting_windows(
    time_s: NDArray[np.float64], nodes: Sequence[float]
) -> tuple[NDArray[np.float64], ...]:
    """Return inclusive windows ending at actual capture samples.

    The final node selects a prefix; the capture may continue beyond it. Nodes
    must exist exactly in the clock. No interpolation, time rounding, or retiming
    is performed. Shared boundary samples occur in both adjacent windows.
    """
    clock, ends = np.asarray(time_s, dtype=float), np.asarray(nodes, dtype=float)
    if (
        clock.ndim != 1
        or clock.size < 2
        or not np.isfinite(clock).all()
        or clock[0] != 0.0
        or np.any(np.diff(clock) <= 0)
        or ends.ndim != 1
        or not ends.size
        or not np.isfinite(ends).all()
        or ends[0] <= 0.0
        or np.any(np.diff(ends) <= 0)
    ):
        raise ValueError("Invalid capture clock or shooting nodes")
    start = 0
    windows = []
    for end in ends:
        matches = np.flatnonzero(clock == end)
        if len(matches) != 1:
            raise ValueError("Shooting node must be an actual capture sample")
        stop = int(matches[0])
        window = clock[start : stop + 1].copy()
        window.setflags(write=False)
        windows.append(window)
        start = stop
    return tuple(windows)
