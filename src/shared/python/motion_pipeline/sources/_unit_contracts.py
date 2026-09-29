"""Shared source-unit and timing contract helpers for mocap adapters."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

from src.shared.python.motion_pipeline.sources.base import (
    AdapterContractError,
    UnitSystem,
)


@dataclass(frozen=True)
class SpatialUnitContract:
    """Validated spatial unit metadata for a source file."""

    token: str
    scale_to_meters: float
    metadata_unit_system: UnitSystem


_SPATIAL_UNITS: dict[str, SpatialUnitContract] = {
    "mm": SpatialUnitContract("mm", 0.001, "millimeters"),
    "millimeter": SpatialUnitContract("millimeter", 0.001, "millimeters"),
    "millimeters": SpatialUnitContract("millimeters", 0.001, "millimeters"),
    "millimetre": SpatialUnitContract("millimetre", 0.001, "millimeters"),
    "millimetres": SpatialUnitContract("millimetres", 0.001, "millimeters"),
    "cm": SpatialUnitContract("cm", 0.01, "centimeters"),
    "centimeter": SpatialUnitContract("centimeter", 0.01, "centimeters"),
    "centimeters": SpatialUnitContract("centimeters", 0.01, "centimeters"),
    "centimetre": SpatialUnitContract("centimetre", 0.01, "centimeters"),
    "centimetres": SpatialUnitContract("centimetres", 0.01, "centimeters"),
    "m": SpatialUnitContract("m", 1.0, "meters"),
    "meter": SpatialUnitContract("meter", 1.0, "meters"),
    "meters": SpatialUnitContract("meters", 1.0, "meters"),
    "metre": SpatialUnitContract("metre", 1.0, "meters"),
    "metres": SpatialUnitContract("metres", 1.0, "meters"),
}

RUST_PRE_SCALED_SAFE_UNITS = frozenset(
    token
    for token, contract in _SPATIAL_UNITS.items()
    if contract.scale_to_meters in {0.001, 1.0}
)


def normalize_spatial_units(
    units: object,
    *,
    format_name: str,
    path: Path,
    default: str = "mm",
) -> SpatialUnitContract:
    """Return a validated spatial-unit contract for a mocap source."""
    token = str(units if units is not None else "").strip().lower() or default
    try:
        return _SPATIAL_UNITS[token]
    except KeyError as exc:
        expected = ", ".join(sorted(_SPATIAL_UNITS))
        raise AdapterContractError(
            f"Unsupported {format_name} units {token!r} in {path}; "
            f"expected one of [{expected}]"
        ) from exc


def require_rust_prescaled_units_are_trusted(
    units: object,
    *,
    format_name: str,
    path: Path,
    default: str = "mm",
) -> SpatialUnitContract:
    """Validate units already pre-scaled by Rust before trusting coordinates."""
    contract = normalize_spatial_units(
        units,
        format_name=format_name,
        path=path,
        default=default,
    )
    if contract.token not in RUST_PRE_SCALED_SAFE_UNITS:
        raise AdapterContractError(
            f"{format_name} Rust backend cannot trust pre-scaled units "
            f"{contract.token!r} in {path}; use the Python parser or update "
            "upstream_mocap_io unit scaling first"
        )
    return contract


def resolve_fps(
    value: object,
    *,
    format_name: str,
    path: Path,
    logger: logging.Logger,
    allow_default: bool,
) -> float:
    """Validate a source frame rate, optionally warning and defaulting to 30 Hz."""
    try:
        fps = float(value)
    except (TypeError, ValueError):
        fps = 0.0
    if math.isfinite(fps) and fps > 0.0:
        return fps
    if allow_default:
        logger.warning(
            "defaulting %s fps to 30.0 for %s because source rate %r is invalid",
            format_name,
            path,
            value,
        )
        return 30.0
    raise AdapterContractError(
        f"{format_name} fps must be finite and > 0 in {path}; got {value!r}"
    )
