"""Data structures and constants for C3D motion-capture processing."""

from __future__ import annotations

from dataclasses import dataclass

C3DMapping = dict

SCHEMA_VERSION = "1.0"

BIOMECHANICAL_MARKER_MIN_M = 0.001
BIOMECHANICAL_MARKER_MAX_M = 10.0


@dataclass(frozen=True)
class C3DEvent:
    """A labeled event occurring at a specific time within a capture."""

    label: str
    time: float

    def __post_init__(self) -> None:
        """Validate event data."""
        if not self.label:
            raise ValueError("Event label cannot be empty.")


@dataclass(frozen=True)
class C3DMetadata:
    """Describes key properties of a C3D motion-capture recording."""

    marker_labels: list[str]
    frame_count: int
    frame_rate: float
    units: str
    analog_labels: list[str]
    analog_units: list[str]
    analog_rate: float | None
    events: list[C3DEvent]

    def __post_init__(self) -> None:
        """Validate metadata fields."""
        if self.frame_count < 0:
            raise ValueError(f"Frame count cannot be negative: {self.frame_count}")
        if self.frame_rate < 0:
            raise ValueError(f"Frame rate cannot be negative: {self.frame_rate}")
        if self.analog_rate is not None and self.analog_rate < 0:
            raise ValueError(f"Analog rate cannot be negative: {self.analog_rate}")

        if len(self.analog_units) != len(self.analog_labels):
            raise ValueError(
                "analog_units and analog_labels must have the same length: "
                f"{len(self.analog_units)} units vs {len(self.analog_labels)} labels"
            )

    @property
    def marker_count(self) -> int:
        """Number of tracked markers in the recording."""

        return len(self.marker_labels)

    @property
    def analog_count(self) -> int:
        """Number of analog channels in the recording."""

        return len(self.analog_labels)

    @property
    def duration(self) -> float:
        """Capture duration in seconds, or ``0`` if the rate is missing."""

        if self.frame_rate == 0:
            return 0.0
        return self.frame_count / self.frame_rate
