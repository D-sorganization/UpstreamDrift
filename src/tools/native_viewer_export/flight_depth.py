"""Scene-depth occlusion of the shot tracer overlay (#15150).

The renderer writes a depth image (metres along the camera axis) so the golfer
and club hide the tracer where they are nearer than it. Pure array functions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

Array = NDArray[np.float64]


def project(camera: Any, points_m: Array, near_clip_m: float) -> tuple[Array, Array]:
    """Pixels and camera depth of world points; ``nan`` pixel nearer than ``near_clip_m``."""
    local = camera.camera_from_world(np.asarray(points_m, dtype=float))
    depth = local[:, 2]
    safe = np.where(depth > near_clip_m, depth, np.nan)
    k = camera.matrix
    u = k[0, 0] * local[:, 0] / safe + k[0, 2]
    v = k[1, 1] * local[:, 1] / safe + k[1, 2]
    return np.column_stack([u, v]), depth


@dataclass(frozen=True)
class Occlusion:
    """Scene depth of one view tile and the slack before it hides a point."""

    depth_m: Array
    tolerance_m: float
    fraction: float
    step_px: float
    near_clip_m: float

    def visible(self, camera: Any, points_m: Array) -> tuple[Array, np.ndarray]:
        """Pixels of ``points_m`` and which the scene does not hide (off-image shows)."""
        pixels, depth = project(camera, points_m, self.near_clip_m)
        height, width = self.depth_m.shape[:2]
        ok = np.isfinite(pixels).all(axis=1)
        u = np.where(ok, np.rint(pixels[:, 0]), -1).astype(int)
        v = np.where(ok, np.rint(pixels[:, 1]), -1).astype(int)
        ok &= (u >= 0) & (u < width) & (v >= 0) & (v < height)
        scene = np.full(len(depth), np.inf)
        scene[ok] = self.depth_m[v[ok], u[ok]]
        slack = self.tolerance_m + self.fraction * depth
        return pixels, np.asarray(scene >= depth - slack, dtype=bool)

    def dense(self, camera: Any, points_m: Array) -> Array:
        """``points_m`` with each segment split into pieces of at most ``step_px``."""
        pts = np.asarray(points_m, dtype=float)
        pix, _ = project(camera, pts, self.near_clip_m)
        length = np.hypot(*np.diff(pix, axis=0).T)
        pieces = np.where(np.isfinite(length), np.ceil(length / self.step_px), 1.0)
        out = [pts[:1]]
        for i, n in enumerate(np.clip(pieces, 1, 4096).astype(int)):
            t = (np.arange(1, n + 1) / n)[:, None]
            out.append(pts[i] + t * (pts[i + 1] - pts[i]))
        return np.vstack(out)


def draw_line(
    pen: ImageDraw.ImageDraw,
    camera: Any,
    points_m: Array,
    colour: tuple[int, ...],
    width_px: int,
    near_clip_m: float,
    occlusion: Occlusion | None,
) -> None:
    """Draw the world polyline ``points_m``, leaving out what the scene hides."""
    if occlusion is None:
        pixels, _ = project(camera, points_m, near_clip_m)
        mask = np.ones(len(pixels), dtype=bool)
    else:
        pixels, mask = occlusion.visible(camera, occlusion.dense(camera, points_m))
    keep = mask & np.isfinite(pixels).all(axis=1)
    edges = np.flatnonzero(np.diff(np.r_[False, keep, False].astype(int)))
    for a, b in zip(edges[::2], edges[1::2], strict=True):
        if b - a > 1:
            pen.line([(float(u), float(v)) for u, v in pixels[a:b]], colour, width_px)


def draw_ball(
    image: Image.Image,
    centre_px: tuple[float, float],
    radius_px: float,
    colour: tuple[int, ...],
    sphere: tuple[float, float],
    occlusion: Occlusion | None,
) -> None:
    """Paint the ball disc, leaving out each pixel the scene is nearer than.

    ``sphere`` is the (camera depth of the centre, radius) in metres; the ball's
    own surface is nearer than its centre, so a scene ball or club touching it
    does not hide it. Without ``occlusion`` the whole disc is painted.
    """
    depth_c, radius_m = sphere
    if not (radius_px > 0.0 and radius_m > 0.0 and depth_c > 0.0):
        raise ValueError("ball radii and depth must be positive")
    u, v = centre_px
    box = (
        max(int(math.floor(u - radius_px)), 0),
        max(int(math.floor(v - radius_px)), 0),
        min(int(math.ceil(u + radius_px)) + 1, image.width),
        min(int(math.ceil(v + radius_px)) + 1, image.height),
    )
    if box[0] >= box[2] or box[1] >= box[3]:
        return
    cols, rows = np.meshgrid(np.arange(box[0], box[2]), np.arange(box[1], box[3]))
    rho2 = ((cols - u) ** 2 + (rows - v) ** 2) / radius_px**2
    inside = rho2 <= 1.0
    if occlusion is not None:
        front = depth_c - radius_m * np.sqrt(np.clip(1.0 - rho2, 0.0, 1.0))
        scene = occlusion.depth_m[box[1] : box[3], box[0] : box[2]]
        slack = occlusion.tolerance_m + occlusion.fraction * front
        inside &= np.asarray(scene >= front - slack, dtype=bool)
    mask = Image.fromarray((inside * 255).astype(np.uint8))
    image.paste(tuple(colour), (box[0], box[1]), mask)


def depth_request(swing: Any, settings: Any, indices: Any) -> list[bool] | None:
    """Positions of ``indices`` from the impact on, or ``None`` when no depth is needed."""
    flight = settings.ball_flight
    if flight is None or not flight.occlude_by_scene:
        return None
    impact = settings.impact_time_s
    return [impact is None or swing.time_s(k) >= impact - 1e-9 for k in indices]
