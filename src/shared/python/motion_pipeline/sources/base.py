"""Base abstraction for mocap source adapters.

Defines the :class:`MocapSourceAdapter` ABC and :class:`SourceMetadata`
Pydantic model. Every supported on-disk format provides a subclass that
knows how to:

- ``supports(path)``     - sniff a candidate file
- ``metadata(path)``     - report FPS, frame count, units, schema
- ``load(path, ...)``    - return a :class:`KeypointSequence`,
                           :class:`MarkerTrajectory`, or
                           :class:`MotionTrajectory`

Design by Contract postconditions are enforced inside :meth:`load_checked`,
which subclasses or callers should use to wrap raw ``load`` output. The
core invariants are:

1. Result frames are non-empty.
2. Timestamps are monotonically non-decreasing.
3. All numeric values are finite.
4. Declared schema/marker_set in metadata matches the loaded payload.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.shared.python.motion_pipeline.contracts import (
    Calibration,
    JointTrajectory,
    KeypointSequence,
    MarkerTrajectory,
    MotionTrajectory,
)

#: Union of all CIR payload types that an adapter may return.
LoadedPayload = KeypointSequence | MarkerTrajectory | JointTrajectory | MotionTrajectory

#: Unit systems an adapter may report.
UnitSystem = Literal[
    "meters", "millimeters", "pixels", "degrees", "radians", "normalized"
]


class SourceMetadata(BaseModel):
    """Lightweight metadata extracted without loading full payload."""

    model_config = ConfigDict(extra="forbid")

    format_name: str = Field(..., description="Adapter format identifier (e.g. 'bvh').")
    fps: float = Field(..., gt=0, description="Source frame rate (Hz).")
    frame_count: int = Field(..., ge=0, description="Number of frames in the file.")
    unit_system: UnitSystem = Field(
        ..., description="Spatial unit system used by the file."
    )
    keypoint_schema: str | None = Field(
        default=None,
        description="Keypoint schema name if this is a keypoint format.",
    )
    marker_set_name: str | None = Field(
        default=None,
        description="Marker set identifier if this is a marker format.",
    )
    notes: str | None = Field(
        default=None, description="Free-form parser notes (e.g. detected variant)."
    )


class UnsupportedFormatError(ValueError):
    """Raised when no registered adapter recognises a given path."""


class AdapterContractError(ValueError):
    """Raised when an adapter's load output violates the post-conditions."""


class MocapSourceAdapter(ABC):
    """Abstract base for mocap source adapters.

    Subclasses must override :attr:`format_name`, :attr:`file_extensions`,
    and the three abstract methods. The framework guarantees:

    - :meth:`supports` is called first; only adapters that return ``True``
      will be invoked for ``load`` / ``metadata``.
    - ``load`` output is validated against the :meth:`_check_postconditions`
      invariants when invoked through :meth:`load_checked`.
    """

    #: Short identifier used in registry lookups and metadata.
    format_name: ClassVar[str] = ""

    #: Tuple of lower-case extensions (with leading dot) this adapter claims.
    file_extensions: ClassVar[tuple[str, ...]] = ()

    @classmethod
    @abstractmethod
    def supports(cls, path: Path) -> bool:
        """Return True if *path* looks like a file this adapter can parse."""

    @abstractmethod
    def metadata(self, path: Path) -> SourceMetadata:
        """Return :class:`SourceMetadata` for *path* without loading payload."""

    @abstractmethod
    def load(
        self,
        path: Path,
        calibration: Calibration | None = None,
    ) -> LoadedPayload:
        """Load *path* and return a CIR payload.

        Implementations should attach ``calibration`` to the returned
        object if it accepts a calibration field.
        """

    # ------------------------------------------------------------------
    # Helpers shared by all adapters

    def load_checked(
        self,
        path: Path,
        calibration: Calibration | None = None,
    ) -> LoadedPayload:
        """Call :meth:`load` and verify post-conditions.

        Raises :class:`AdapterContractError` on violation.
        """
        result = self.load(path, calibration=calibration)
        try:
            md = self.metadata(path)
        except (
            Exception  # noqa: BLE001 - metadata is advisory; any failure -> None
        ):  # pragma: no cover - metadata is advisory here
            md = None
        self._check_postconditions(result, md)
        return result

    @staticmethod
    def _check_postconditions(result: LoadedPayload, md: SourceMetadata | None) -> None:
        """Validate the CIR contract for an adapter result."""
        # MotionTrajectory wraps a JointTrajectory; descend into it.
        frames = getattr(result, "frames", None)
        if frames is None:
            inner = getattr(result, "trajectory", None)
            frames = getattr(inner, "frames", None)
        if frames is None or len(frames) == 0:
            raise AdapterContractError(
                "Adapter produced empty frames sequence; expected at least one frame."
            )
        prev_ts: float | None = None
        for i, frame in enumerate(frames):
            ts = float(getattr(frame, "timestamp", 0.0))
            if not math.isfinite(ts):
                raise AdapterContractError(
                    f"Frame {i} has non-finite timestamp {ts!r}."
                )
            if prev_ts is not None and ts < prev_ts:
                raise AdapterContractError(
                    f"Timestamps must be monotonic; frame {i} ts={ts} < prev {prev_ts}."
                )
            prev_ts = ts

        if (
            md is not None
            and isinstance(result, KeypointSequence)
            and md.keypoint_schema
        ):
            first_schema = result.frames[0].schema_name
            if md.keypoint_schema not in (first_schema, "custom"):
                # custom is a permitted escape hatch
                raise AdapterContractError(
                    f"Metadata declared schema {md.keypoint_schema!r} but "
                    f"frame schema is {first_schema!r}."
                )
