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


def export_markers_to_c3d(
    markers: np.ndarray,
    marker_names: list[str],
    sample_rate: float,
    output_path: str | Path,
    *,
    up_axis: str = "z",
    units: str = "m",
) -> Path:
    """Export marker position data to a C3D file.

    This is a **one-way, output-only** export.  The resulting file is
    intended for external biomechanical tools and must never be read back
    into the pipeline as a canonical data source.

    Preconditions:
        - ``markers`` is a ``numpy.ndarray`` with shape ``(T, n_markers, 3)``
          where T >= 1 and n_markers >= 1.
        - ``len(marker_names) == markers.shape[1]``
        - ``sample_rate > 0.0``
        - ``up_axis`` is either ``"y"`` or ``"z"``.
        - ``units`` is either ``"m"`` or ``"mm"``.
        - Parent directory of ``output_path`` must exist.
        - ``ezc3d >= 1.5.0`` must be installed.

    Postconditions:
        - A valid C3D file is written to ``output_path``.
        - The returned path is absolute (resolved).
        - Marker labels, positions, units, and sample rate are
          preserved across a write / read round-trip.
        - The exported data is in canonical Z-up convention.
        - Homogeneous 4th point row is 1.0.
        - Residuals in ``c['data']['meta_points']['residuals']`` are 0.0 for
          valid points and -1.0 for NaN/occluded points.

    Args:
        markers: Marker positions in metres, shape ``(T, n_markers, 3)``.
        marker_names: Ordered list of marker label strings.
            ``len(marker_names)`` must equal ``markers.shape[1]``.
        sample_rate: Acquisition rate in Hz.  Must be strictly positive.
        output_path: Destination file path.  Parent directory must exist.
        up_axis: Up-axis convention of input markers (``"y"`` or ``"z"``).
            If ``"y"`` (ADR-0041), markers are converted to canonical Z-up.
            Defaults to ``"z"``.
        units: Destination unit string (``"m"`` or ``"mm"``). Defaults to ``"m"``.

    Returns:
        Resolved :class:`~pathlib.Path` of the written C3D file.

    Raises:
        TypeError: When ``markers`` is not a :class:`numpy.ndarray`.
        ValueError: When shape, rate, up_axis, units, or name-count constraints
            are violated.
        FileNotFoundError: When the parent directory of ``output_path`` does
            not exist.
        ImportError: When ``ezc3d`` is not installed.
    """
    # --- DbC preconditions (validated before the optional ezc3d import) ---
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

    norm_up_axis = up_axis.lower()
    norm_units = units.lower()

    dest = Path(output_path)
    if not dest.parent.exists():
        raise FileNotFoundError(f"Output directory does not exist: {dest.parent}")

    # --- Optional dependency: ezc3d ---
    try:
        ezc3d = import_ezc3d()
    except ModuleNotFoundError as exc:
        raise ImportError(
            "ezc3d is required for C3D export. "
            "Install it with: pip install 'upstream-drift[c3d]' "
            "(or: pip install 'ezc3d>=1.5.0')"
        ) from exc

    # Transform coordinates to canonical Z-up if provided as Y-up (ADR-0041)
    coords = markers.astype(np.float64)
    if norm_up_axis == "y":
        coords = coords @ _ADR0041_TO_CANONICAL.T

    # Scale to requested units (input markers are in metres)
    if norm_units == "mm":
        coords = coords * 1000.0

    # Build the C3D object
    c = ezc3d.c3d()
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

    # Detect NaN values across coordinates (shape: T, n_markers)
    is_nan = np.isnan(markers).any(axis=-1)

    # ezc3d stores points as (4, n_markers, T): rows are [X, Y, Z, homogeneous_scale].
    # Row 3 is homogeneous coordinate (1.0).
    data_xyz = np.transpose(coords, (2, 1, 0))  # (3, n_markers, T)
    homogeneous = np.ones((1, n_markers, n_frames), dtype=np.float64)
    c["data"]["points"] = np.concatenate([data_xyz, homogeneous], axis=0)

    # Residuals live in c["data"]["meta_points"]["residuals"], shape (1, n_markers, T).
    # C3D standard encodes valid points as 0.0 and missing/occluded/NaN as -1.0.
    residuals = np.where(is_nan.T[np.newaxis, :, :], -1.0, 0.0).astype(np.float64)
    if "meta_points" not in c["data"]:
        c["data"]["meta_points"] = {}
    c["data"]["meta_points"]["residuals"] = residuals

    c.write(str(dest))
    logger.info("Exported %d-frame C3D (%d markers) → %s", n_frames, n_markers, dest)
    return dest.resolve()
