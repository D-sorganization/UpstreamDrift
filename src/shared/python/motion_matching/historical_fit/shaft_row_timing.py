"""Fixed row-time shaft diagnostics, never a sensor calibration or optimizer.

Caller context is a declaration. A persistence/worker boundary must authenticate
the canonical capture, decoded pixels, source clock and native motion first.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
import re
from typing import TYPE_CHECKING

import numpy as np

from .contracts import CameraProjection, ImageFitResult
from .shaft_residuals import (
    ImageResidualAssessment,
    ShaftAxisResidualTerm,
    project_authored_shaft_line,
)

if TYPE_CHECKING:
    from src.shared.python.motion_matching.pipeline.plant import MatchingPlant


def _finite_real(value: object) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and bool(np.isfinite(value))
    )


@dataclass(frozen=True)
class ShaftRowTiming:
    """Explicit encoded-raster assumptions with unknown physical timing.

    t(y) = t_PTS + direction * readout * (y/(height-1) - reference).
    Cropped/sensor/field mappings are deliberately unsupported. The fixed saved
    camera is reused; no time-varying camera motion is inferred.
    """

    source_clock_sha256: str
    evidence_sha256: str
    readout_source_seconds: float
    scan_direction: int
    reference_row_fraction: float = 0.5
    row_mapping: str = "authored_full_encoded_raster"

    def __post_init__(self) -> None:
        for value in (self.source_clock_sha256, self.evidence_sha256):
            if not isinstance(value, str) or not re.fullmatch(
                r"sha256:[0-9a-f]{64}", value
            ):
                raise ValueError(
                    "Row timing requires prefixed lowercase identity digests"
                )
        if (
            not _finite_real(self.readout_source_seconds)
            or self.readout_source_seconds < 0
        ):
            raise ValueError(
                "Readout requires finite nonnegative encoded-clock seconds"
            )
        if type(self.scan_direction) is not int or self.scan_direction not in (-1, 1):
            raise ValueError("Scan direction requires integer -1 or 1")
        if (
            not _finite_real(self.reference_row_fraction)
            or not 0 <= self.reference_row_fraction <= 1
        ):
            raise ValueError("Reference row fraction requires a finite value in [0,1]")
        if self.row_mapping != "authored_full_encoded_raster":
            raise ValueError(
                "Only explicit authored full encoded raster mapping is supported"
            )

    @property
    def physical_time_qualified(self) -> bool:
        """Authored encoded-clock timing cannot qualify the physical clock."""
        return False


@dataclass(frozen=True, kw_only=True)
class RowTimedShaftAssessment(ImageResidualAssessment):
    """Endpoint distances at separate times; no common straight bearing angle."""

    row_timing: ShaftRowTiming
    endpoint_source_times: tuple[tuple[float, float] | None, ...]
    perpendicular_errors_pixels: tuple[tuple[float, float] | None, ...]

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            not isinstance(self.row_timing, ShaftRowTiming)
            or self.row_timing.evidence_sha256 != self.evidence_sha256
        ):
            raise ValueError("Assessment row timing differs from evidence identity")
        times = tuple(
            None if row is None else tuple(row) for row in self.endpoint_source_times
        )
        errors = tuple(
            None if row is None else tuple(row)
            for row in self.perpendicular_errors_pixels
        )
        if len(times) != len(self.frame_indices) or len(errors) != len(times):
            raise ValueError("Endpoint rows must align with all source frames")
        for row, error in zip(times, errors, strict=True):
            if (row is None) != (error is None):
                raise ValueError("Abstentions require absent times and errors")
            if (
                row is not None
                and error is not None
                and (
                    len(row) != 2
                    or len(error) != 2
                    or not all(_finite_real(x) for x in (*row, *error))
                )
            ):
                raise ValueError("Endpoint times and errors require finite pairs")
        observed = [row for row in errors if row is not None]
        rms = float(np.sqrt(np.mean(np.square(observed)))) if observed else None
        if (rms is None) != (self.raw_rms_pixels is None) or (
            rms is not None
            and not np.isclose(rms, self.raw_rms_pixels, rtol=1e-12, atol=1e-12)
        ):
            raise ValueError("Raw RMS differs from all unweighted observed endpoints")
        object.__setattr__(self, "endpoint_source_times", times)
        object.__setattr__(self, "perpendicular_errors_pixels", errors)


def _endpoint_times(
    term: ShaftAxisResidualTerm, timing: ShaftRowTiming
) -> tuple[tuple[float, float] | None, ...]:
    height = term.evidence.image_size[1]
    if height < 2:
        raise ValueError("Encoded raster row mapping requires height at least two")
    rows: list[tuple[float, float] | None] = []
    for time, frame in zip(term.source_times, term.evidence.frames, strict=True):
        if frame.segment.points_px is None:
            rows.append(None)
            continue
        shifted = [
            time
            + timing.scan_direction
            * timing.readout_source_seconds
            * (point[1] / (height - 1) - timing.reference_row_fraction)
            for point in frame.segment.points_px
        ]
        rows.append((shifted[0], shifted[1]))
    return tuple(rows)


def assess_row_timed_shaft(
    native: MatchingPlant,
    camera: CameraProjection,
    motion: ImageFitResult,
    term: ShaftAxisResidualTerm,
    timing: ShaftRowTiming,
) -> RowTimedShaftAssessment:
    """Assess a fixed motion and saved camera after whole-role support checking.

    Reject every case with any unsupported endpoint before evaluating motion or
    native FK. Never clamp, extrapolate or omit endpoints. Zero readout delegates
    to the unchanged legacy assessment for exact numerical parity. This pure
    diagnostic authenticates no source, estimates no readout, and runs no fit.
    """
    if (
        not isinstance(motion, ImageFitResult)
        or not isinstance(term, ShaftAxisResidualTerm)
        or not isinstance(timing, ShaftRowTiming)
    ):
        raise ValueError("Typed motion, shaft term and row timing required")
    if (timing.source_clock_sha256, timing.evidence_sha256) != (
        term.source_clock_sha256,
        term.evidence.sha256,
    ):
        raise ValueError("Row timing source clock or evidence identity differs")
    if (
        motion.model_sha != native.plant_sha
        or motion.coordinate_order != native.coordinate_order
        or term.axis.native_model_sha != native.plant_sha
    ):
        raise ValueError(
            "Motion/axis model identity or native coordinate order differs"
        )
    rows = _endpoint_times(term, timing)
    flat_times = np.asarray([time for row in rows if row is not None for time in row])
    if np.any(flat_times < motion.knot_times[0]) or np.any(
        flat_times > motion.knot_times[-1]
    ):
        raise ValueError(
            "Complete evidence role requires all endpoint times inside spline support"
        )
    errors: tuple[tuple[float, float] | None, ...]
    if timing.readout_source_seconds == 0:
        legacy = term.assess(
            native, camera, motion.evaluate_source_times(np.asarray(term.source_times))
        )
        errors, rms = legacy.perpendicular_errors_pixels, legacy.raw_rms_pixels
    else:
        poses = (
            iter(motion.evaluate_source_times(flat_times))
            if len(flat_times)
            else iter(())
        )
        values: list[tuple[float, float] | None] = []
        for frame in term.evidence.frames:
            if frame.segment.points_px is None:
                values.append(None)
                continue
            pair = []
            for point in frame.segment.points_px:
                origin, normal, _, _ = project_authored_shaft_line(
                    native, camera, term.axis, next(poses)
                )
                pair.append(float((np.asarray(point) - origin) @ normal))
            values.append((pair[0], pair[1]))
        errors = tuple(values)
        observed = [row for row in errors if row is not None]
        rms = float(np.sqrt(np.mean(np.square(observed)))) if observed else None
    return RowTimedShaftAssessment(
        evidence_sha256=term.evidence.sha256,
        frame_indices=tuple(frame.frame_index for frame in term.evidence.frames),
        source_times=term.source_times,
        raw_rms_pixels=rms,
        row_timing=timing,
        endpoint_source_times=rows,
        perpendicular_errors_pixels=errors,
    )
