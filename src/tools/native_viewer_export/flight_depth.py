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


#: the near-plane cut is placed this fraction in front of the plane, so it projects
_NEAR_MARGIN = 1e-3


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


def clip_near(camera: Any, points_m: Array, near_clip_m: float) -> Array:
    """``points_m`` with a point added where each segment crosses the near plane.

    A segment from in front of the camera to behind it then keeps its visible
    part (an aim line whose finish lies behind a face-on camera, #15150).
    """
    pts = np.asarray(points_m, dtype=float)
    depth = camera.camera_from_world(pts)[:, 2] - near_clip_m * (1.0 + _NEAR_MARGIN)
    out = [pts[:1]]
    for i in range(1, len(pts)):
        if (depth[i - 1] > 0.0) != (depth[i] > 0.0):
            t = depth[i - 1] / (depth[i - 1] - depth[i])
            out.append(pts[i - 1 : i] + t * (pts[i : i + 1] - pts[i - 1 : i]))
        out.append(pts[i : i + 1])
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
    points_m = clip_near(camera, points_m, near_clip_m)
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


@dataclass(frozen=True)
class BallShading:
    """Sphere shading of the ball disc (meanings: ``flight_tracer_settings.json``)."""

    shaded: bool
    light_dir: tuple[float, ...]
    ambient: float
    highlight_strength: float
    highlight_size: float

    def __post_init__(self) -> None:
        if not isinstance(self.shaded, bool):
            raise TypeError("ball_shading shaded must be true or false")
        light = tuple(float(c) for c in self.light_dir)
        if len(light) != 3 or not all(math.isfinite(c) for c in light):
            raise ValueError("ball_shading light_dir must be 3 finite numbers")
        if math.sqrt(sum(c * c for c in light)) <= 0.0:
            raise ValueError("ball_shading light_dir must not be the zero vector")
        object.__setattr__(self, "light_dir", light)
        for name in ("ambient", "highlight_strength"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"ball_shading {name} must lie in 0..1")
        if not 0.0 < self.highlight_size <= 1.0:
            raise ValueError("ball_shading highlight_size must lie in (0, 1]")


def _shade(
    colour: tuple[int, ...],
    shading: BallShading,
    nx: NDArray[Any],
    ny: NDArray[Any],
    nz: NDArray[Any],
) -> NDArray[np.uint8]:
    """RGB (rows, cols, 3) of a sphere of ``colour`` with unit normals ``n``."""
    light = np.array(shading.light_dir) / np.linalg.norm(shading.light_dir)
    lambert = np.clip(nx * light[0] + ny * light[1] + nz * light[2], 0.0, None)
    level = shading.ambient + (1.0 - shading.ambient) * lambert
    half = light + np.array([0.0, 0.0, 1.0])  # viewer looks along -z
    half /= np.linalg.norm(half)
    angle = np.arccos(np.clip(nx * half[0] + ny * half[1] + nz * half[2], -1.0, 1.0))
    glint = np.clip(1.0 - angle / (shading.highlight_size * math.pi / 2), 0.0, 1.0)
    rgb = np.asarray(colour[:3], dtype=float) * level[..., None]
    rgb += 255.0 * shading.highlight_strength * glint[..., None] ** 2
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def draw_ball(
    image: Image.Image,
    centre_px: tuple[float, float],
    radius_px: float,
    colour: tuple[int, ...],
    sphere: tuple[float, float],
    occlusion: Occlusion | None,
    shading: BallShading | None = None,
) -> None:
    """Paint the ball disc, leaving out each pixel the scene is nearer than.

    ``sphere`` is the (camera depth of the centre, radius) in metres; the ball's
    own surface is nearer than its centre, so a scene ball or club touching it
    does not hide it. Without ``occlusion`` the whole disc is painted. A
    ``shading`` that is on lights the disc as a sphere; the covered pixels are
    the same either way.
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
    fill: Any = tuple(colour)
    if shading is not None and shading.shaded:
        nx, ny = (cols - u) / radius_px, (rows - v) / radius_px
        nz = np.sqrt(np.clip(1.0 - rho2, 0.0, 1.0))
        fill = Image.fromarray(_shade(colour, shading, nx, ny, nz), "RGB")
    image.paste(fill, (box[0], box[1]), mask)


def depth_request(swing: Any, settings: Any, indices: Any) -> list[bool] | None:
    """Positions of ``indices`` from the impact on, or ``None`` when no depth is needed."""
    flight = settings.ball_flight
    if flight is None or not flight.occlude_by_scene:
        return None
    impact = settings.impact_time_s
    return [impact is None or swing.time_s(k) >= impact - 1e-9 for k in indices]
