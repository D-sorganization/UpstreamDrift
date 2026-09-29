"""JSON-serializable plot data structures for the analysis service layer.

These dataclasses are the structured, frontend-agnostic representation of
analysis/plot results (issue #7446).  They contain only plain Python types
(``str``, ``float``, ``list``, ``dict``) so that they can be serialized to
JSON / wrapped in Pydantic response models without further conversion, and
rendered by either the PyQt6 (matplotlib) frontend or the web frontend.

This module must stay free of Qt and matplotlib imports.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = ["CounterfactualResult", "PlotData", "PlotSeries"]


@dataclass
class PlotSeries:
    """A single named data series within a plot.

    Attributes:
        name: Human-readable series label (legend entry).
        x: X-axis values (commonly time in seconds, but may be a spatial
            coordinate for trajectory plots).
        y: Y-axis values, same length as ``x``.
        z: Optional Z-axis values for 3D trajectory plots.
        units: Units of the y values (e.g. ``"deg"``, ``"W"``, ``"m"``).
        metadata: Extra series-level information (peak values, indices...).

    Postcondition: ``to_dict()`` output is JSON-serializable.
    """

    name: str
    x: list[float]
    y: list[float]
    z: list[float] | None = None
    units: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("series name must be a string")
        if len(self.x) != len(self.y):
            raise ValueError(
                f"series '{self.name}': x and y lengths differ "
                f"({len(self.x)} != {len(self.y)})"
            )
        if self.z is not None and len(self.z) != len(self.x):
            raise ValueError(
                f"series '{self.name}': z length {len(self.z)} does not "
                f"match x length {len(self.x)}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation."""
        return asdict(self)


@dataclass
class PlotData:
    """Structured, renderer-agnostic description of a plot.

    Attributes:
        plot_type: Registry identifier (e.g. ``"joint_angles"``).
        title: Plot title.
        x_label: X axis label (including units).
        y_label: Y axis label (including units).
        series: The data series to draw.
        metadata: Plot-level information (joint indices, messages, ...).

    Postcondition: ``to_dict()`` output is JSON-serializable.
    """

    plot_type: str
    title: str
    x_label: str
    y_label: str
    series: list[PlotSeries] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.plot_type, str) or not self.plot_type:
            raise ValueError("plot_type must be a non-empty string")

    @property
    def is_empty(self) -> bool:
        """True when no series carry data (e.g. nothing was recorded)."""
        return all(len(s.x) == 0 for s in self.series)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation."""
        return asdict(self)


@dataclass
class CounterfactualResult:
    """Result of a counterfactual / induced-acceleration computation.

    Attributes:
        kind: Computation kind (``"ztcf"``, ``"zvcf"``, ``"gravity"``,
            ``"drift"``, ``"control"``, ``"total"``).
        times: Time stamps in seconds, length ``n_frames``.
        values: Per-frame vectors (``n_frames x n_dofs``).
        units: Units of ``values`` (joint accelerations: ``"rad/s^2"``).
        metadata: Extra information (frame count, source, ...).

    Postcondition: ``to_dict()`` output is JSON-serializable.
    """

    kind: str
    times: list[float]
    values: list[list[float]]
    units: str = "rad/s^2"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("kind must be a non-empty string")
        if len(self.times) != len(self.values):
            raise ValueError(
                f"times and values lengths differ "
                f"({len(self.times)} != {len(self.values)})"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation."""
        return asdict(self)
