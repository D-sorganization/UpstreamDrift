"""Core types, exceptions, and constants for programmatic-pid.

Design-by-Contract: This module defines the shared vocabulary used across
all other modules. No module-level side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple


class SpecValidationError(ValueError):
    """Raised when a YAML specification fails contract validation."""


@dataclass
class ValidationIssue:
    """Structured validation error for agent-friendly consumption.

    Attributes:
        path: Dot-separated path into the spec (e.g. 'equipment[2].id').
        message: Human-readable description.
        severity: 'error' or 'warning'.
    """

    path: str
    message: str
    severity: str = "error"

    def to_dict(self) -> dict[str, str]:
        """Serialize to a plain dict suitable for JSON output."""
        return {"path": self.path, "message": self.message, "severity": self.severity}


class Point(NamedTuple):
    """2-D coordinate."""

    x: float
    y: float


class BBox(NamedTuple):
    """Axis-aligned bounding box (x_min, y_min, x_max, y_max)."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def center(self) -> Point:
        return Point(
            (self.x_min + self.x_max) / 2.0,
            (self.y_min + self.y_max) / 2.0,
        )

    def contains_point(self, p: Point) -> bool:
        return self.x_min <= p.x <= self.x_max and self.y_min <= p.y <= self.y_max

    def overlaps(self, other: BBox, pad: float = 0.0) -> bool:
        return not (
            self.x_max + pad <= other.x_min
            or other.x_max + pad <= self.x_min
            or self.y_max + pad <= other.y_min
            or other.y_max + pad <= self.y_min
        )

    def union(self, other: BBox) -> BBox:
        return BBox(
            min(self.x_min, other.x_min),
            min(self.y_min, other.y_min),
            max(self.x_max, other.x_max),
            max(self.y_max, other.y_max),
        )

    def expanded(self, margin: float) -> BBox:
        return BBox(
            self.x_min - margin,
            self.y_min - margin,
            self.x_max + margin,
            self.y_max + margin,
        )


@dataclass
class LayoutRegions:
    """Result of layout computation.

    Invariant: panels never overlap equipment_bbox.
    """

    layout_cfg: dict[str, Any]
    equipment_bbox: BBox
    canvas_bbox: BBox
    panels: dict[str, tuple[float, float, float, float]]


@dataclass
class TextConfig:
    """Text sizing configuration derived from spec."""

    title_height: float
    subtitle_height: float
    body_height: float
    small_height: float


# Type alias for the raw parsed spec dictionary.
SpecDict = dict[str, Any]
