"""Framing of the OpenSim simbody view (#12308), pure numpy (no OpenSim import).

The simbody window (800x600 under xvfb, with a toolbar strip at its bottom) is
not the clip's 16:9, so the legacy centre-crop dropped part of the vertical
field of view and pushed the feet below the image. The worker now crops about
the camera's principal point, sets the simbody field of view so that the
visible crop has the shared viewer lens (``VIEWER_FOV_Y_RAD``), and fits the
camera distance to the trajectory's foot, head and clubhead points.
"""

from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np
from numpy.typing import NDArray

from src.shared.python.golf_view_presets import (
    DEFAULT_FRAME_MARGIN,
    fit_distance_m,
    get_view_preset,
)

#: body names whose origins bound the golfer in the face-on framing
FRAMING_BODY_PREFIXES = ("calcn", "toes")
FRAMING_BODY_NAMES = ("Head", "Clubhead")


def principal_crop(
    window_hw: tuple[int, int], strip_px: int, size_px: tuple[int, int]
) -> tuple[int, int, int, int]:
    """``(y0, y1, x0, x1)`` of the largest output-aspect crop about the window centre.

    ``window_hw`` is the grabbed window ``(height, width)``; the bottom
    ``strip_px`` rows are the visualizer toolbar and stay outside the crop.
    Postcondition: the crop is centred on the principal point, so the look-at
    point projects to the centre of the output image.
    """
    h_w, w_w = window_hw
    width, height = size_px
    if min(h_w, w_w, width, height) < 1 or strip_px < 0:
        raise ValueError("window, size and strip must be positive")
    aspect = width / height
    half_h = min(h_w / 2.0 - strip_px, w_w / 2.0 / aspect)
    crop_h = int(2 * half_h) // 2 * 2
    if crop_h < 2:
        raise ValueError("window too small for the output aspect")
    crop_w = min(w_w, int(round(crop_h * aspect)))
    y0, x0 = (h_w - crop_h) // 2, (w_w - crop_w) // 2
    return y0, y0 + crop_h, x0, x0 + crop_w


def simbody_fov_y_rad(target_fov_y_rad: float, window_h: int, crop_h: int) -> float:
    """Simbody vertical field of view whose ``crop_h`` rows span ``target_fov_y_rad``."""
    if not 0.0 < target_fov_y_rad < math.pi:
        raise ValueError("target_fov_y_rad must be in (0, pi)")
    if not 0 < crop_h <= window_h:
        raise ValueError("crop_h must be in (0, window_h]")
    return 2.0 * math.atan(math.tan(target_fov_y_rad / 2.0) * window_h / crop_h)


def framing_body_names(names: Sequence[str]) -> list[str]:
    """Bodies bounding the framing (feet, head, clubhead), in ``names`` order."""
    return [
        n
        for n in names
        if n in FRAMING_BODY_NAMES or n.startswith(FRAMING_BODY_PREFIXES)
    ]


def sole_points(
    points_m: NDArray[np.float64], ground_height_m: float
) -> NDArray[np.float64]:
    """``points_m`` plus the same points dropped to the ground (foot soles)."""
    pts = np.asarray(points_m, dtype=float).reshape(-1, 3)
    dropped = pts.copy()
    dropped[:, 2] = ground_height_m
    return np.vstack([pts, dropped])


def fitted_distance_m(
    view: str,
    points_m: NDArray[np.float64] | Sequence[Sequence[float]],
    lookat_m: Sequence[float],
    distance_m: float | None,
    fov_y_rad: float,
    aspect: float,
    margin: float = DEFAULT_FRAME_MARGIN,
) -> float:
    """Camera distance for ``view``: an explicit ``distance_m``, else a fit.

    The fit is the preset's default distance, or the distance at which
    ``points_m`` leave ``margin`` empty border if that is larger: never closer
    than the shared preset.  Raises ``ValueError`` for invalid points.
    """
    if distance_m is not None:
        return float(distance_m)
    preset = get_view_preset(view)
    fit = fit_distance_m(
        preset, points_m, lookat_m, fov_y_rad=fov_y_rad, aspect=aspect, margin=margin
    )
    return max(preset.default_distance_m, fit)


def fitted_lookat_m(
    view: str,
    points_m: NDArray[np.float64] | Sequence[Sequence[float]],
    lookat_m: Sequence[float],
) -> list[float]:
    """Look-at point moved in the image plane to the centre of ``points_m``.

    The golfer, club swing included, is not centred on the preset's static
    look-at (a driver's follow-through reaches 2.7 m), so a camera aimed at the
    box centre needs a shorter distance to keep everything in frame.  The depth
    coordinate is unchanged.  Raises ``ValueError`` for invalid points.
    """
    pts = np.asarray(points_m, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 1 or pts.shape[1] != 3:
        raise ValueError("points_m must be a non-empty (n, 3) array")
    preset = get_view_preset(view)
    look = np.asarray(lookat_m, dtype=float)
    for axis in (preset.image_right(), preset.image_up()):
        proj = pts @ axis
        look = look + axis * (0.5 * (proj.min() + proj.max()) - look @ axis)
    return [float(v) for v in look]
