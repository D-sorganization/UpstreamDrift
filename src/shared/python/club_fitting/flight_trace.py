"""Ball-flight series of one flex in the target frame (#15150 C5).

Downrange along the target line, side right-positive (the flight model's
``lateral_m`` sign) and height up, metres from the tee. The sweep's shot model
returns it under :data:`FLIGHT_KEY`; ``flex_sweep.json`` stores it under
``"flights"`` and the flex comparison overlays it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

FLIGHT_KEY = "flight"
_KEYS = ("downrange_m", "side_m", "height_m")


@dataclass(frozen=True, eq=False)
class FlightTrace:
    """Downrange, side (+ right) and height (m) of the ball after impact."""

    downrange_m: np.ndarray
    side_m: np.ndarray
    height_m: np.ndarray

    def __post_init__(self) -> None:
        arrays = [np.asarray(getattr(self, k), dtype=float) for k in _KEYS]
        if len({a.shape for a in arrays}) != 1 or arrays[0].ndim != 1:
            raise ValueError("downrange_m, side_m and height_m must be the same length")
        if arrays[0].size < 2 or not np.all(np.isfinite(arrays)):
            raise ValueError("a flight trace needs two or more finite points")
        for name, a in zip(_KEYS, arrays, strict=True):
            object.__setattr__(self, name, a)

    def to_dict(self) -> dict[str, list[float]]:
        return {k: np.asarray(getattr(self, k), dtype=float).tolist() for k in _KEYS}

    @classmethod
    def from_dict(cls, data: dict[str, list[float]]) -> FlightTrace:
        """Inverse of :meth:`to_dict`. Raises ``ValueError`` on a bad shape."""
        try:
            return cls(*(np.asarray(data[k]) for k in _KEYS))
        except KeyError as exc:
            raise ValueError(f"flight trace is missing {exc}") from exc


def flight_from_playback(play: Any) -> FlightTrace | None:
    """Target-frame trace of a ``FlightPlayback``; ``None`` without a path.

    ``axes_m`` rows are the target, left and up axes, so side is minus left.
    """
    if play.path_m is None or play.axes_m is None:
        return None
    path = np.asarray(play.path_m, dtype=float)
    local = (path - path[0]) @ np.asarray(play.axes_m, dtype=float).T
    return FlightTrace(local[:, 0], -local[:, 1], local[:, 2])
