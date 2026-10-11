from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class C3DEvent:
    """A named C3D event with its timestamp in seconds."""

    label: str
    time: float


@dataclass
class MarkerData:
    """Represents a single optical marker trajectory."""

    name: str
    position: npt.NDArray[np.float64]  # shape (N, 3)
    residuals: npt.NDArray[np.float64] | None = None


@dataclass
class AnalogData:
    """Represents a single analog channel (e.g., EMG, force plate)."""

    name: str
    values: npt.NDArray[np.float64]  # shape (N,)
    unit: str = ""


@dataclass
class C3DDataModel:
    """Aggregated data model for a loaded C3D file."""

    filepath: str
    markers: dict[str, MarkerData] = field(default_factory=dict)
    analog: dict[str, AnalogData] = field(default_factory=dict)
    point_rate: float = 0.0
    analog_rate: float = 0.0
    point_time: npt.NDArray[np.float64] | None = None
    analog_time: npt.NDArray[np.float64] | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    events: list[C3DEvent] = field(default_factory=list)
    raw_parameters: dict | None = None
    force_series: Any | None = None
    force_missing: tuple[str, ...] = ()

    def marker_names(self) -> list[str]:
        """Return list of marker names."""
        return list(self.markers.keys())

    def analog_names(self) -> list[str]:
        """Return list of analog channel names."""
        return list(self.analog.keys())
