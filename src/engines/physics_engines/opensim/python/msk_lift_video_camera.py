"""Lift video camera, schedule, and dynamic framing (OSL-8, #13045)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Final

import numpy as np

from src.motion_capture.reconstruct.cameras import PinholeCamera
from src.shared.python.contracts import require
from src.shared.python.video_timing.frame_schedule import FrameSchedule

LIFT_VIEWS: Final[tuple[str, ...]] = ("side", "front", "oblique")
LIFT_OVERLAYS: Final[tuple[str, ...]] = ("forces", "clean", "none")
BARBELL_LIFTS: Final[tuple[str, ...]] = (
    "squat",
    "deadlift",
    "snatch",
    "clean_and_jerk",
    "bench_press",
)
VIEWER_FOV_Y_RAD: Final[float] = math.radians(45.0)
UI_STRIP_PX: Final[int] = 40
SETTLE_S: Final[float] = 0.5


@dataclass(frozen=True)
class LiftVideoConfig:
    """Configuration for headless OpenSim lift video export."""

    lift: str
    view: str = "side"
    fps: int = 60
    speed: float = 1.0
    overlay: str = "forces"
    width: int = 800
    height: int = 600

    def __post_init__(self) -> None:
        if self.view not in LIFT_VIEWS:
            raise ValueError(f"unknown view {self.view!r}; choose from {LIFT_VIEWS}")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.speed <= 0.0:
            raise ValueError(f"speed must be positive, got {self.speed}")
        if self.overlay not in LIFT_OVERLAYS:
            raise ValueError(
                f"unknown overlay {self.overlay!r}; choose from {LIFT_OVERLAYS}"
            )
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")


def lift_camera_transforms(
    view: str,
    lookat_m: Sequence[float] | tuple[float, float, float],
    distance_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute camera rotation matrices and position for lifting views.

    Lifting uses Y-up world coordinates (+Y up, +X anterior/forward, +Z right).
    Returns (simbody_rot, pinhole_rot, cam_pos) where both rotations are orthonormal
    with det +1, and the pinhole camera projects lookat_m to (0, 0, Z_cam) with Z_cam > 0.
    """
    lookat = np.asarray(lookat_m, dtype=np.float64)
    require(lookat.shape == (3,), "lookat_m must be a 3-vector")
    require(distance_m > 0.0, "distance_m must be positive")

    if view == "side":
        # 30 deg front-oblique off sagittal axis so 0.45m plates do not occlude body (OSL-8, #13045)
        angle = math.radians(30.0)
        view_dir = np.array([-math.sin(angle), 0.0, -math.cos(angle)])
        up = np.array([0.0, 1.0, 0.0])
        right = np.array([math.cos(angle), 0.0, -math.sin(angle)])
    elif view == "front":
        # Looking from anterior towards -X: lifter's left foot (-Z) is to the right
        view_dir = np.array([-1.0, 0.0, 0.0])
        up = np.array([0.0, 1.0, 0.0])
        right = np.array([0.0, 0.0, -1.0])
    elif view == "oblique":
        # Angled 45 deg between front and right side
        view_dir = np.array([-1.0, 0.0, -1.0]) / math.sqrt(2.0)
        up = np.array([0.0, 1.0, 0.0])
        right = np.array([1.0, 0.0, -1.0]) / math.sqrt(2.0)
    else:
        raise ValueError(f"unknown view {view!r}; choose from {LIFT_VIEWS}")

    cam_pos = lookat - view_dir * float(distance_m)

    # Simbody visualizer camera: cols = [image_right, image_up, -view_dir]
    simbody_rot = np.column_stack([right, up, -view_dir])

    # Pinhole camera (OpenCV / graphics convention): cols = [image_right, -image_up, view_dir]
    pinhole_rot = np.column_stack([right, -up, view_dir])

    return simbody_rot, pinhole_rot, cam_pos


def pinhole_for_lifting_view(
    view: str,
    lookat_m: Sequence[float],
    distance_m: float,
    fov_y_rad: float,
    size_px: tuple[int, int],
) -> PinholeCamera:
    """Build a PinholeCamera for lifting view matching the simbody visualizer."""
    width, height = size_px
    require(width > 0 and height > 0, "size_px must be positive")
    require(0.0 < fov_y_rad < math.pi, "fov_y_rad must be in (0, pi)")

    _, pinhole_rot, cam_pos = lift_camera_transforms(view, lookat_m, distance_m)
    focal = (height / 2.0) / math.tan(fov_y_rad / 2.0)
    matrix = np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]]
    )
    return PinholeCamera(
        camera_id=f"lift_{view}",
        matrix=matrix,
        rotation_world_from_camera=pinhole_rot,
        translation_world_from_camera_m=cam_pos,
        image_size_px=(width, height),
    )


def lift_video_schedule(
    times_s: np.ndarray,
    q: np.ndarray,
    fps: int = 60,
    speed: float = 1.0,
) -> tuple[FrameSchedule, np.ndarray, np.ndarray]:
    """Compute frame schedule, interpolated coordinates, and nearest solve indices."""
    sched = FrameSchedule(times_s, fps=float(fps), speed=float(speed))
    q_interp = sched.interpolate(q)
    nearest = sched.nearest_indices()
    return sched, q_interp, nearest


def compute_lift_framing(
    positions_m: np.ndarray,
    bar_endpoints_m: np.ndarray | None = None,
    view: str = "side",
    fov_y_rad: float = VIEWER_FOV_Y_RAD,
    image_size_px: tuple[int, int] = (800, 600),
    margin: float = 0.08,
    min_lifter_height_ratio: float = 0.60,
) -> tuple[tuple[float, float, float], float]:
    """Compute camera lookat and distance to dynamically frame lifter and barbell."""
    pts_list = [positions_m.reshape(-1, 3)]
    if bar_endpoints_m is not None:
        pts_list.append(bar_endpoints_m.reshape(-1, 3))
    all_pts = np.vstack(pts_list)
    p_min = all_pts.min(axis=0)
    p_max = all_pts.max(axis=0)
    lookat = (
        float((p_min[0] + p_max[0]) / 2.0),
        float((p_min[1] + p_max[1]) / 2.0),
        float((p_min[2] + p_max[2]) / 2.0),
    )

    _, pinhole_rot, _ = lift_camera_transforms(view, lookat, 1.0)
    w, h = image_size_px
    aspect = w / h
    half_fov_y = fov_y_rad / 2.0
    tan_y = math.tan(half_fov_y)
    tan_x = aspect * tan_y

    rel = all_pts - np.asarray(lookat)
    cam_pts = rel @ pinhole_rot

    x_cam = np.abs(cam_pts[:, 0])
    y_cam = np.abs(cam_pts[:, 1])
    z_cam = cam_pts[:, 2]

    eff_margin = max(0.01, min(0.40, margin))
    d_x = x_cam / ((1.0 - eff_margin) * tan_x) - z_cam
    d_y = y_cam / ((1.0 - eff_margin) * tan_y) - z_cam
    d_min = max(float(np.max(d_x)), float(np.max(d_y)))

    lifter_pts = positions_m.reshape(-1, 3)
    lifter_h = float(lifter_pts[:, 1].max() - lifter_pts[:, 1].min())
    d_max_ratio = lifter_h / (2.0 * min_lifter_height_ratio * tan_y)
    distance = float(np.clip(d_min, 1.5, max(d_min, d_max_ratio)))
    return lookat, distance
