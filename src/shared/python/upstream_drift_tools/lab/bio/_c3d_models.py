from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

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
class ForcePlateCalibration:
    """Calibration data for a single force plate from the C3D FORCE_PLATFORM group.

    Attributes:
        corners: Plate corners in lab frame, shape ``(4, 3)`` in metres. Order is
            +x+y, -x+y, -x-y, +x-y as defined by the C3D specification.
        origin: Origin of the plate measurement coordinate system, expressed as
            an offset from the plate centre. Shape ``(3,)`` in metres.
        cal_matrix: Calibration matrix relating raw analog voltages to
            forces/moments. Shape varies by plate type (None for type 1, where
            channels are pre-calibrated). Type 2/3: 6x6. Type 4: 6x6 typical.
            Type 7/11/12: larger.
        plate_type: C3D plate type code (1-4 supported in this reader). 1 means
            channels already report forces/moments in plate frame; 2/3/4 require
            ``cal_matrix`` to be applied to raw voltages.
        channel_indices: ``(start, end)`` indices into the analog channel array
            (inclusive start, exclusive end) that belong to this plate.
    """

    corners: np.ndarray
    origin: np.ndarray
    cal_matrix: np.ndarray | None
    plate_type: int
    channel_indices: tuple[int, int]

    def __post_init__(self) -> None:
        """Validate calibration shapes."""
        if self.corners.shape != (4, 3):
            raise ValueError(
                f"corners must have shape (4, 3); got {self.corners.shape}"
            )
        if self.origin.shape != (3,):
            raise ValueError(f"origin must have shape (3,); got {self.origin.shape}")
        if self.plate_type not in (1, 2, 3, 4):
            raise ValueError(f"plate_type must be one of 1-4; got {self.plate_type}")
        start, end = self.channel_indices
        if start < 0 or end < start:
            raise ValueError(
                f"channel_indices must be (start, end) with 0 <= start <= end; "
                f"got {self.channel_indices}"
            )


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
    force_plates: tuple[ForcePlateCalibration, ...] = field(default_factory=tuple)

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
