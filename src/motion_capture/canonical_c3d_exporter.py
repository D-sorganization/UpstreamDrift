"""Canonical-state → C3D exporter (CC-16).

**OUTPUT-ONLY** — this module is a terminal export for downstream
biomechanical tools (e.g. Visual3D, OpenSim, Mokka).  C3D must never
be used as a lossy intermediate format inside the pipeline.

Architecture invariant: ``canonical_c3d_exporter`` must not be imported
by simulation-pipeline internals (``trace_io``, ``protocol``,
``pose_interchange``, etc.).  This constraint is asserted by the
architecture test in
``tests/unit/motion_capture/test_canonical_c3d_exporter.py``.

Typical usage::

    from src.motion_capture.canonical_c3d_exporter import export_markers_to_c3d

    # markers from Trace.markers or from FK on canonical sites
    export_markers_to_c3d(
        markers=trace.markers,   # shape (T, n_markers, 3) [m]
        marker_names=site_names,
        sample_rate=200.0,
        output_path="results/swing_export.c3d",
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.data_io.native_dll_guard import import_ezc3d

__all__ = ["export_markers_to_c3d"]

logger = logging.getLogger(__name__)


_CANONICAL_TO_ADR0041 = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)
_ADR0041_TO_CANONICAL = _CANONICAL_TO_ADR0041.T


def _validate_c3d_export_preconditions(
    markers: np.ndarray,
    marker_names: list[str],
    sample_rate: float,
    up_axis: str,
    units: str,
    output_path: str | Path,
) -> Path:
    """Validate all DbC preconditions before C3D export."""
    if not isinstance(markers, np.ndarray):
        raise TypeError(
            f"markers must be a numpy.ndarray, got {type(markers).__name__!r}"
        )
    if markers.ndim != 3 or markers.shape[2] != 3:
        raise ValueError(
            f"markers must have shape (T, n_markers, 3), got {markers.shape!r}"
        )
    n_frames, n_markers, _ = markers.shape
    if n_frames < 1:
        raise ValueError(
            f"markers must contain at least one frame (T >= 1), got T={n_frames}"
        )
    if n_markers < 1:
        raise ValueError(
            f"markers must contain at least one marker (n_markers >= 1), "
            f"got n_markers={n_markers}"
        )
    if len(marker_names) != n_markers:
        raise ValueError(
            f"len(marker_names)={len(marker_names)!r} does not match "
            f"markers.shape[1]={n_markers!r}"
        )
    if not isinstance(sample_rate, (int, float)) or sample_rate <= 0.0:
        raise ValueError(f"sample_rate must be a positive number, got {sample_rate!r}")
    if not isinstance(up_axis, str) or up_axis.lower() not in ("y", "z"):
        raise ValueError(f"up_axis must be 'y' or 'z', got {up_axis!r}")
    if not isinstance(units, str) or units.lower() not in ("m", "mm"):
        raise ValueError(f"units must be 'm' or 'mm', got {units!r}")

    dest = Path(output_path)
    if not dest.parent.exists():
        raise FileNotFoundError(f"Output directory does not exist: {dest.parent}")
    return dest


def _populate_c3d_data(
    c: Any,
    coords: np.ndarray,
    is_nan: np.ndarray,
    marker_names: list[str],
    sample_rate: float,
    norm_units: str,
    up_axis: str,
) -> None:
    """Populate parameters, point arrays, and metadata residuals in an ezc3d object."""
    n_frames, n_markers, _ = coords.shape
    c["parameters"]["POINT"]["RATE"]["value"] = [float(sample_rate)]
    c["parameters"]["POINT"]["LABELS"]["value"] = list(marker_names)
    c["parameters"]["POINT"]["UNITS"]["value"] = [norm_units]

    if "DESCRIPTIONS" not in c["parameters"]["POINT"]:
        c["parameters"]["POINT"]["DESCRIPTIONS"] = {}
    c["parameters"]["POINT"]["DESCRIPTIONS"]["value"] = [
        f"up_axis=Z (source={up_axis.upper()})" for _ in marker_names
    ]

    if "AXIS_CONVENTION" not in c["parameters"]:
        c["parameters"]["AXIS_CONVENTION"] = {}
    if "UP_AXIS" not in c["parameters"]["AXIS_CONVENTION"]:
        c["parameters"]["AXIS_CONVENTION"]["UP_AXIS"] = {}
    c["parameters"]["AXIS_CONVENTION"]["UP_AXIS"]["value"] = ["Z"]

    # Point data with homogeneous 4th row
    data_xyz = np.transpose(coords, (2, 1, 0))  # (3, n_markers, T)
    homogeneous = np.ones((1, n_markers, n_frames), dtype=np.float64)
    c["data"]["points"] = np.concatenate([data_xyz, homogeneous], axis=0)

    # Residuals: valid = 0.0, NaN/missing = -1.0
    residuals = np.where(is_nan.T[np.newaxis, :, :], -1.0, 0.0).astype(np.float64)
    if "meta_points" not in c["data"]:
        c["data"]["meta_points"] = {}
    c["data"]["meta_points"]["residuals"] = residuals


def export_markers_to_c3d(
    markers: np.ndarray,
    marker_names: list[str],
    sample_rate: float,
    output_path: str | Path,
    *,
    up_axis: str = "z",
    units: str = "m",
) -> Path:
    """Export marker position data to a C3D file."""
    dest = _validate_c3d_export_preconditions(
        markers, marker_names, sample_rate, up_axis, units, output_path
    )
    norm_up_axis = up_axis.lower()
    norm_units = units.lower()

    try:
        ezc3d = import_ezc3d()
    except ModuleNotFoundError as exc:
        raise ImportError(
            "ezc3d is required for C3D export. "
            "Install it with: pip install 'upstream-drift[c3d]' "
            "(or: pip install 'ezc3d>=1.5.0')"
        ) from exc

    coords = markers.astype(np.float64)
    if norm_up_axis == "y":
        coords = coords @ _ADR0041_TO_CANONICAL.T
    if norm_units == "mm":
        coords = coords * 1000.0

    is_nan = np.isnan(markers).any(axis=-1)
    c = ezc3d.c3d()
    _populate_c3d_data(
        c, coords, is_nan, marker_names, sample_rate, norm_units, up_axis
    )

    c.write(str(dest))
    logger.info(
        "Exported %d-frame C3D (%d markers) → %s",
        markers.shape[0],
        markers.shape[1],
        dest,
    )
    return dest.resolve()
