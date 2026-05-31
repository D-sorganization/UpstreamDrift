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

__all__ = ["export_markers_to_c3d"]

logger = logging.getLogger(__name__)


def export_markers_to_c3d(
    markers: np.ndarray,
    marker_names: list[str],
    sample_rate: float,
    output_path: str | Path,
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
        - Parent directory of ``output_path`` must exist.
        - ``ezc3d >= 1.5.0`` must be installed.

    Postconditions:
        - A valid C3D file is written to ``output_path``.
        - The returned path is absolute (resolved).
        - Marker labels, positions, units (``"m"``), and sample rate are
          preserved across a write / read round-trip.

    Args:
        markers: Marker positions in metres, shape ``(T, n_markers, 3)``.
        marker_names: Ordered list of marker label strings.
            ``len(marker_names)`` must equal ``markers.shape[1]``.
        sample_rate: Acquisition rate in Hz.  Must be strictly positive.
        output_path: Destination file path.  Parent directory must exist.

    Returns:
        Resolved :class:`~pathlib.Path` of the written C3D file.

    Raises:
        TypeError: When ``markers`` is not a :class:`numpy.ndarray`.
        ValueError: When shape, rate, or name-count constraints are violated.
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

    dest = Path(output_path)
    if not dest.parent.exists():
        raise FileNotFoundError(f"Output directory does not exist: {dest.parent}")

    # --- Optional dependency: ezc3d ---
    try:
        import ezc3d  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "ezc3d is required for C3D export. "
            "Install it with: pip install 'ezc3d>=1.5.0'"
        ) from exc

    # Build the C3D object
    c = ezc3d.c3d()
    c["parameters"]["POINT"]["RATE"]["value"] = [float(sample_rate)]
    c["parameters"]["POINT"]["LABELS"]["value"] = list(marker_names)
    c["parameters"]["POINT"]["UNITS"]["value"] = ["m"]

    # ezc3d stores data as (4, n_markers, T): rows are [X, Y, Z, residual].
    # Transpose from (T, n_markers, 3) → (3, n_markers, T), then append
    # a zero-residual row to make it (4, n_markers, T).
    data_xyz = np.transpose(markers.astype(np.float64), (2, 1, 0))  # (3, n_markers, T)
    residuals = np.zeros((1, n_markers, n_frames), dtype=np.float64)
    c["data"]["points"] = np.concatenate([data_xyz, residuals], axis=0)

    c.write(str(dest))
    logger.info("Exported %d-frame C3D (%d markers) → %s", n_frames, n_markers, dest)
    return dest.resolve()
