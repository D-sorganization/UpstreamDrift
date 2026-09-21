"""Immutable, serializable calibration receipt for swing planes and geometry (TB-03 #10588).

Captures all calibration parameters, coordinate transformations, residuals, and frozen
inertia contracts in a tamper-evident, versioned dataclass.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

CALIBRATION_RECEIPT_SCHEMA_VERSION = "tour-calibration-receipt/1.0.0"


@dataclass(frozen=True)
class CalibrationReceipt:
    """Immutable record of calibrated swing plane, geometry, and initial state."""

    capture_filename: str
    capture_sha256: str
    model_name: str
    sample_rate_hz: float
    window_start_s: float
    window_end_s: float
    plane_origin: tuple[float, float, float]
    plane_normal: tuple[float, float, float]
    plane_u_axis: tuple[float, float, float]
    plane_v_axis: tuple[float, float, float]
    plane_inclination_deg: float
    gravity_in_plane: tuple[float, float, float]
    arm_length_m: float
    club_length_m: float
    arm_length_std_m: float
    club_length_std_m: float
    shaft_mass_kg: float
    clubhead_mass_kg: float
    arm_mass_kg: float
    initial_theta1_rad: float
    initial_theta2_rad: float
    initial_omega1_rad_s: float
    initial_omega2_rad_s: float
    fk_closure_residual_m: float
    plane_fit_rmse_m: float
    plane_fit_max_residual_m: float
    hub_condition: str = "fixed_pivot"
    external_work_joules: float = 0.0
    is_mass_frozen: bool = True
    is_inertia_frozen: bool = True
    schema_version: str = CALIBRATION_RECEIPT_SCHEMA_VERSION
    marker_offsets: dict[str, tuple[float, float, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert receipt to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize receipt to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationReceipt:
        """Deserialize receipt from dictionary."""
        d = dict(data)
        schema = d.get("schema_version")
        if schema != CALIBRATION_RECEIPT_SCHEMA_VERSION:
            logger.warning(
                "CalibrationReceipt schema %s differs from expected %s",
                schema,
                CALIBRATION_RECEIPT_SCHEMA_VERSION,
            )
        # Convert tuples
        for k in (
            "plane_origin",
            "plane_normal",
            "plane_u_axis",
            "plane_v_axis",
            "gravity_in_plane",
        ):
            if k in d and isinstance(d[k], (list, tuple)):
                d[k] = (float(d[k][0]), float(d[k][1]), float(d[k][2]))
        if "marker_offsets" in d and isinstance(d["marker_offsets"], dict):
            d["marker_offsets"] = {
                name: (float(vec[0]), float(vec[1]), float(vec[2]))
                for name, vec in d["marker_offsets"].items()
            }
        return cls(**d)

    @classmethod
    def from_json(cls, json_str: str) -> CalibrationReceipt:
        """Deserialize receipt from JSON string."""
        return cls.from_dict(json.loads(json_str))
