"""Club calibration specification and cross-contamination validation (PF-02, #10432).

Enforces separate Driver and 7-Iron calibration models, shaft length consistency,
marker geometry attachments, and guards against silent cross-contamination.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import logging
from typing import Any

import numpy as np

from src.shared.python.contracts import ensure, require
from src.shared.python.motion_matching.club_models import (
    DRIVER,
    IRON_7,
    ClubSpec,
)

logger = logging.getLogger(__name__)


class ClubCompatibilityError(ValueError):
    """Raised when an incompatible club calibration is used on geometry or data."""


class ClubType(str, enum.Enum):
    """Supported club categories for kinematic calibration."""

    DRIVER = "driver"
    IRON_7 = "iron7"
    WOOD_3 = "wood3"
    HYBRID = "hybrid"
    WEDGE = "wedge"


@dataclass(frozen=True)
class ClubCalibrationSpec:
    """Rigid club calibration containing physical properties and marker attachments."""

    club_type: ClubType
    spec: ClubSpec
    marker_attachments: dict[str, tuple[float, float, float]]
    expected_shaft_length_range_m: tuple[float, float]
    lie_angle_deg: float
    loft_angle_deg: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require(
            self.expected_shaft_length_range_m[0]
            <= self.spec.length_m
            <= self.expected_shaft_length_range_m[1],
            "spec length must lie within expected range",
        )
        require(
            len(self.marker_attachments) >= 2,
            "at least two marker attachments required for rigid registration",
        )


def get_driver_calibration_spec() -> ClubCalibrationSpec:
    """Return default rigid calibration spec for driver."""
    markers = {
        "Marker_Club_1": (0.010, -0.100, 0.064),
        "Marker_Club_2": (0.050, 0.000, 0.064),
        "Marker_Club_3": (-0.050, 0.000, 0.064),
        "Marker_Club_4": (0.000, -0.050, 0.100),
    }
    return ClubCalibrationSpec(
        club_type=ClubType.DRIVER,
        spec=DRIVER,
        marker_attachments=markers,
        expected_shaft_length_range_m=(1.050, 1.250),
        lie_angle_deg=58.0,
        loft_angle_deg=10.5,
        metadata={"model": "Tour Driver 45.5in", "shaft": "graphite_stiff"},
    )


def get_iron_7_calibration_spec() -> ClubCalibrationSpec:
    """Return default rigid calibration spec for 7-iron."""
    markers = {
        "Marker_Club_1": (0.005, -0.080, 0.064),
        "Marker_Club_2": (0.025, 0.000, 0.064),
        "Marker_Club_3": (-0.025, 0.000, 0.064),
        "Marker_Club_4": (0.000, -0.040, 0.080),
    }
    return ClubCalibrationSpec(
        club_type=ClubType.IRON_7,
        spec=IRON_7,
        marker_attachments=markers,
        expected_shaft_length_range_m=(0.880, 0.990),
        lie_angle_deg=62.5,
        loft_angle_deg=34.0,
        metadata={"model": "Tour 7-Iron 37.0in", "shaft": "steel_regular"},
    )


def validate_club_compatibility(
    calibration: ClubCalibrationSpec,
    target_club_type: ClubType | str,
    *,
    target_shaft_length_m: float | None = None,
) -> None:
    """Validate that calibration matches target club type and physical dimensions.

    Raises:
        ClubCompatibilityError: If calibration club type or dimensions do not match target.
    """
    target_val = (
        target_club_type.value
        if isinstance(target_club_type, ClubType)
        else str(target_club_type)
    )
    target_str = target_val.lower()
    c_type = calibration.club_type
    calib_val = c_type.value
    calib_str = calib_val.lower()

    if calib_str != target_str:
        msg = (
            f"Incompatible club calibration: cannot apply {calib_str} calibration "
            f"to {target_str} geometry or capture"
        )
        logger.error(msg)
        raise ClubCompatibilityError(msg)

    if target_shaft_length_m is not None:
        lo, hi = calibration.expected_shaft_length_range_m
        if not (lo <= target_shaft_length_m <= hi):
            msg = (
                f"Shaft length {target_shaft_length_m:.3f} m is outside valid calibration "
                f"range [{lo:.3f}, {hi:.3f}] m for {calib_str}"
            )
            logger.error(msg)
            raise ClubCompatibilityError(msg)


def diagnose_club_marker_residuals(
    calibration: ClubCalibrationSpec,
    measured_marker_offsets: Mapping[str, Sequence[float]],
    *,
    tolerance_m: float = 0.025,
) -> dict[str, Any]:
    """Diagnose marker-by-marker residuals against calibrated club model.

    Returns:
        Dictionary containing marker residuals, RMS error, and compatibility status.
    """
    residuals: dict[str, float] = {}
    errors_sq: list[float] = []

    for name, calib_pos in calibration.marker_attachments.items():
        if name in measured_marker_offsets:
            meas_pos = np.asarray(measured_marker_offsets[name], dtype=float)
            cal_arr = np.asarray(calib_pos, dtype=float)
            err = float(np.linalg.norm(meas_pos - cal_arr))
            residuals[name] = err
            errors_sq.append(err**2)

    rms = float(np.sqrt(np.mean(errors_sq))) if errors_sq else 0.0
    max_err = max(residuals.values()) if residuals else 0.0
    compatible = (rms <= tolerance_m) and (max_err <= 2.0 * tolerance_m)

    return {
        "club_type": calibration.club_type.value,
        "matched_marker_count": len(residuals),
        "marker_residuals_m": residuals,
        "rms_error_m": rms,
        "max_error_m": max_err,
        "compatible": compatible,
    }
