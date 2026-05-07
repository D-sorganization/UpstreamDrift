"""3D Meshcat overlay for measured + simulated club skeletons.

Implements the Pinocchio engine-bespoke 3D viewer per issue #4133 — both
measured and simulated club skeletons are drawn in different colours
(measured = blue, simulated = red, per VISUALIZATION_SPEC.md).

This intentionally **does not** depend on the larger
:class:`motion_training.motion_visualizer.MotionVisualizer` (which is a
heavyweight humanoid+club viewer) — instead it offers a minimal direct
Meshcat path that can be invoked headlessly when a pre-existing
``MotionVisualizer`` is not available, and a thin :func:`overlay`
extension when one is.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .._types import ClubTargetLike, FitResult

logger = logging.getLogger(__name__)


def _build_polyline_geometry(points: np.ndarray, colour_hex: int):
    """Build a Meshcat ``LineBasicMaterial`` polyline for ``points`` (N,3)."""
    import meshcat.geometry as mcg  # type: ignore[import-not-found]

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"polyline points must be (N,3), got {points.shape}")
    return mcg.Line(
        mcg.PointsGeometry(points.T.astype(np.float32)),
        mcg.LineBasicMaterial(color=colour_hex, linewidth=3),
    )


def meshcat_overlay(
    target: ClubTargetLike,
    result: FitResult,
    *,
    visualizer: Any | None = None,
) -> str:
    """Push measured and simulated clubhead traces into a Meshcat viewer.

    Args:
        target: Measured trajectory; the clubhead path is drawn in blue.
        result: Pinocchio fit result; ``clubhead_sim`` is drawn in red.
        visualizer: An existing Meshcat ``Visualizer`` (or ``MotionVisualizer``
            with a ``viewer`` attribute). If ``None``, a fresh Meshcat viewer
            is constructed.

    Returns:
        The Meshcat URL the user can open to inspect the overlay.

    Raises:
        ImportError: meshcat is not installed.
    """
    import meshcat  # type: ignore[import-not-found]

    if visualizer is None:
        viewer = meshcat.Visualizer()
    else:
        # Accept either a raw meshcat.Visualizer or a MotionVisualizer.
        viewer = getattr(visualizer, "viewer", visualizer)

    measured_pts = np.asarray(target.clubhead, dtype=np.float64)
    simulated_pts = np.asarray(result.clubhead_sim, dtype=np.float64)

    if measured_pts.size:
        viewer["overlay/measured_path"].set_object(
            _build_polyline_geometry(measured_pts, 0x1F77B4)
        )
    if simulated_pts.size:
        viewer["overlay/simulated_path"].set_object(
            _build_polyline_geometry(simulated_pts, 0xD62728)
        )

    url = viewer.url() if hasattr(viewer, "url") else ""
    logger.info("Meshcat overlay published: %s", url)
    return url
