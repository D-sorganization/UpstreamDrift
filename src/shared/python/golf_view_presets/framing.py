"""Projected framing of a golf view preset (NV-9, #11697).

Every native viewer uses one vertical field of view, ``VIEWER_FOV_Y_RAD``, so
a preset and distance frame the golfer the same way in every backend. The
helpers measure how much of the frame a set of world points fills and choose
the camera distance that frames them with a given margin.

The *extent* of a set of points is the largest normalised image coordinate,
``max(|x| / half_width, |y| / half_height)`` about the image centre: ``1`` is
the frame edge, so a margin ``m`` means an extent of ``1 - m``.
"""

from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np
from numpy.typing import NDArray

from .presets import ViewPreset, check_point3

Array = NDArray[np.float64]

#: Vertical field of view of every native viewer camera (OpenSim's 0.7 rad).
VIEWER_FOV_Y_RAD = 0.7
#: Largest empty border, as a fraction of the half frame, along the fitted axis.
DEFAULT_FRAME_MARGIN = 0.15
DEFAULT_ASPECT = 16.0 / 9.0
_FIT_RELATIVE_TOLERANCE = 1e-6
_MAX_BISECTIONS = 200


def _check_points(points_m: Sequence[Sequence[float]] | Array) -> Array:
    pts = np.asarray(points_m, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 1 or pts.shape[1] != 3:
        raise ValueError("points_m must be a non-empty (n, 3) array")
    if not np.isfinite(pts).all():
        raise ValueError("points_m must be finite")
    return pts


def _check_lens(fov_y_rad: float, aspect: float) -> None:
    if not (math.isfinite(fov_y_rad) and 0.0 < fov_y_rad < math.pi):
        raise ValueError(f"fov_y_rad must lie in (0, pi), got {fov_y_rad}")
    if not (math.isfinite(aspect) and aspect > 0.0):
        raise ValueError(f"aspect must be positive and finite, got {aspect}")


def _extent(
    preset: ViewPreset,
    pts: Array,
    look: Array,
    distance_m: float,
    tan_half: float,
    aspect: float,
) -> float:
    rel = pts - preset.camera_position(look, distance_m)
    depth = rel @ preset.view_direction()
    if not (depth > 0.0).all():
        raise ValueError("every point must lie in front of the camera")
    x = np.abs(rel @ preset.image_right()) / (depth * tan_half * aspect)
    y = np.abs(rel @ preset.image_up()) / (depth * tan_half)
    return float(np.max(np.maximum(x, y)))


def projected_extent(
    preset: ViewPreset,
    points_m: Sequence[Sequence[float]] | Array,
    lookat_m: Sequence[float] | Array,
    distance_m: float,
    *,
    fov_y_rad: float = VIEWER_FOV_Y_RAD,
    aspect: float = DEFAULT_ASPECT,
) -> float:
    """Largest normalised image coordinate of ``points_m`` (``1`` = frame edge).

    ``aspect`` is width over height. Raises ``ValueError`` for empty or
    non-finite points, an invalid lens, or a point behind the camera.
    """
    _check_lens(fov_y_rad, aspect)
    look = check_point3(lookat_m, "lookat_m")
    tan_half = math.tan(fov_y_rad / 2.0)
    return _extent(preset, _check_points(points_m), look, distance_m, tan_half, aspect)


def fit_distance_m(
    preset: ViewPreset,
    points_m: Sequence[Sequence[float]] | Array,
    lookat_m: Sequence[float] | Array,
    *,
    fov_y_rad: float = VIEWER_FOV_Y_RAD,
    aspect: float = DEFAULT_ASPECT,
    margin: float = DEFAULT_FRAME_MARGIN,
) -> float:
    """Camera distance at which ``points_m`` reach an extent of ``1 - margin``.

    Postcondition: ``projected_extent`` at the returned distance equals
    ``1 - margin`` to a relative tolerance of 1e-6, so the points fit inside
    the frame with an empty border of at most ``margin``. Raises
    ``ValueError`` when ``margin`` is outside ``[0, 1)`` or the inputs are
    invalid (see :func:`projected_extent`).
    """
    if not (math.isfinite(margin) and 0.0 <= margin < 1.0):
        raise ValueError(f"margin must lie in [0, 1), got {margin}")
    _check_lens(fov_y_rad, aspect)
    pts = _check_points(points_m)
    look = check_point3(lookat_m, "lookat_m")
    tan_half = math.tan(fov_y_rad / 2.0)
    target = 1.0 - margin
    # The nearest admissible camera keeps every point in front of it.
    ahead = float(np.max((look - pts) @ preset.view_direction()))
    near = max(ahead, 0.0) * (1.0 + _FIT_RELATIVE_TOLERANCE) + 1e-9
    far = max(2.0 * near, 1.0)
    while _extent(preset, pts, look, far, tan_half, aspect) > target:
        far *= 2.0
    for _ in range(_MAX_BISECTIONS):
        mid = 0.5 * (near + far)
        if _extent(preset, pts, look, mid, tan_half, aspect) > target:
            near = mid
        else:
            far = mid
        if far - near <= _FIT_RELATIVE_TOLERANCE * far:
            break
    return far
