"""Headless OpenSim barbell lift video rendering and motion adapter (OSL-8, #13045).

Renders barbell lifts (squat, deadlift, snatch, clean and jerk) using OpenSim's
Simbody visualizer under xvfb, featuring:
1. Dynamic per-frame muscle activation recoloring from blue (0.0) to red (1.0).
2. Projected 2D force overlays for per-foot ground reaction forces (GRF) and
   per-hand barbell grip wrenches using PinholeCamera and OpenCV glyphs.
3. Multi-view camera framing (side, front, oblique) aligned to Y-up MSK coords.
4. Deterministic 60 fps frame schedule with 1.0x and 0.5x playback speeds.
5. Honest reporting for bench press (unavailable due to unphysical posture).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Final

import numpy as np

from src.engines.physics_engines.opensim.python.musculoskeletal_swing import (
    resolve_base_model,
)
from src.motion_capture.reconstruct.cameras import PinholeCamera
from src.shared.python.contracts import require
from src.shared.python.engine_core.opensim_runtime import import_opensim
from src.shared.python.force_overlay.contracts import (
    ForceTorqueFrame,
    OverlayWrench,
    WrenchKind,
)
from src.shared.python.force_overlay.glyphs import ForceGlyphStyle, build_glyphs
from src.shared.python.lifting.bundle_targets import LiftTargets, lift_targets
from src.shared.python.lifting.bundles import SAME_INPUT_LIFT_BUNDLES
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.video_timing.frame_schedule import FrameSchedule
from src.tools.native_viewer_export.backends._opensim_framing import (
    principal_crop,
    simbody_fov_y_rad,
)
from src.tools.native_viewer_export.backends._scene import fit_to_size
from src.tools.native_viewer_export.overlay2d import draw_glyphs_rgb

logger = get_logger(__name__)

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
        # Looking from right side towards -Z: lifter faces right (+X)
        view_dir = np.array([0.0, 0.0, -1.0])
        up = np.array([0.0, 1.0, 0.0])
        right = np.array([1.0, 0.0, 0.0])
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


def muscle_color_rgb(activation: float) -> tuple[float, float, float]:
    """Color a muscle by activation from blue (0.0) to red (1.0)."""
    val = float(np.clip(activation, 0.0, 1.0))
    return (val, 0.0, 1.0 - val)


def _vec3(arr: Any) -> tuple[float, float, float]:
    """Convert a 3-sequence to a typed float 3-tuple."""
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def extract_lift_wrenches(targets: Any, frame_idx: int) -> list[OverlayWrench]:
    """Extract 4 OverlayWrenches (2 foot GRF, 2 hand bar wrenches) for frame_idx."""
    lms = targets.landmarks(frame_idx)
    mid_foot_l = _vec3((lms["heel_l"] + lms["forefoot_l"]) / 2.0)
    mid_foot_r = _vec3((lms["heel_r"] + lms["forefoot_r"]) / 2.0)

    grf_l_f = _vec3(targets.grf_force_n["left"][frame_idx])
    grf_r_f = _vec3(targets.grf_force_n["right"][frame_idx])

    grip_l_p = _vec3(targets.grip_m["L"][frame_idx])
    grip_r_p = _vec3(targets.grip_m["R"][frame_idx])

    grip_l_f = _vec3(targets.hand_force_n["L"][frame_idx])
    grip_r_f = _vec3(targets.hand_force_n["R"][frame_idx])

    grip_l_tau = _vec3(targets.hand_torque_nm["L"][frame_idx])
    grip_r_tau = _vec3(targets.hand_torque_nm["R"][frame_idx])

    return [
        OverlayWrench(
            kind=WrenchKind.CONTACT,
            label="contact:grf_left",
            body="calcn_l",
            point_m=mid_foot_l,
            force_n=grf_l_f,
            source="opensim_lift",
        ),
        OverlayWrench(
            kind=WrenchKind.CONTACT,
            label="contact:grf_right",
            body="calcn_r",
            point_m=mid_foot_r,
            force_n=grf_r_f,
            source="opensim_lift",
        ),
        OverlayWrench(
            kind=WrenchKind.GRIP,
            label="grip:hand_l",
            body="hand_l",
            point_m=grip_l_p,
            force_n=grip_l_f,
            torque_nm=grip_l_tau,
            source="opensim_lift",
        ),
        OverlayWrench(
            kind=WrenchKind.GRIP,
            label="grip:hand_r",
            body="hand_r",
            point_m=grip_r_p,
            force_n=grip_r_f,
            torque_nm=grip_r_tau,
            source="opensim_lift",
        ),
    ]


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
    """Compute camera lookat and distance to dynamically frame lifter and barbell.

    Ensures:
    1. Lifter span >= min_lifter_height_ratio of image height when standing/extended.
    2. All lifter points and barbell endpoints fit inside the frame with safety margin.
    3. Lookat is centered at the 3D bounding box center of the movement.
    """
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

    # Lifter vertical height
    lifter_pts = positions_m.reshape(-1, 3)
    lifter_h = float(lifter_pts[:, 1].max() - lifter_pts[:, 1].min())
    d_max_ratio = lifter_h / (2.0 * min_lifter_height_ratio * tan_y)
    distance = float(np.clip(d_min, 1.5, max(d_min, d_max_ratio)))
    return lookat, distance


def _draw_legend_arrows_and_scale(
    out: np.ndarray,
    bx: int,
    by: int,
    font: int,
    scale: float,
    color_text: tuple[int, int, int],
    force_scale_n: float,
    force_scale_px: int,
) -> None:
    import cv2

    # GRF (Green)
    cv2.circle(out, (bx + 18, by + 40), 5, (60, 220, 60), -1, cv2.LINE_AA)
    cv2.putText(
        out,
        "GRF contact force",
        (bx + 32, by + 44),
        font,
        scale,
        color_text,
        1,
        cv2.LINE_AA,
    )

    # Grip (Blue)
    cv2.circle(out, (bx + 18, by + 60), 5, (60, 160, 255), -1, cv2.LINE_AA)
    cv2.putText(
        out,
        "Hand grip wrench",
        (bx + 32, by + 64),
        font,
        scale,
        color_text,
        1,
        cv2.LINE_AA,
    )

    # Force scale bar
    sx0, sy0 = bx + 18, by + 82
    sx1 = sx0 + int(force_scale_px)
    cv2.line(out, (sx0, sy0), (sx1, sy0), (220, 220, 220), 2, cv2.LINE_AA)
    cv2.line(out, (sx0, sy0 - 3), (sx0, sy0 + 3), (220, 220, 220), 1, cv2.LINE_AA)
    cv2.line(out, (sx1, sy0 - 3), (sx1, sy0 + 3), (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(
        out,
        f"{int(force_scale_n)} N scale",
        (sx1 + 10, sy0 + 4),
        font,
        scale,
        color_text,
        1,
        cv2.LINE_AA,
    )


def _draw_legend_activation_bar(
    out: np.ndarray,
    bx: int,
    by: int,
    font: int,
    color_dim: tuple[int, int, int],
) -> None:
    import cv2

    ax0, ay0, aw, ah = bx + 18, by + 102, 130, 10
    grad = np.zeros((ah, aw, 3), dtype=np.uint8)
    for c in range(aw):
        val = c / max(1.0, float(aw - 1))
        grad[:, c, 0] = int(val * 255)
        grad[:, c, 1] = 0
        grad[:, c, 2] = int((1.0 - val) * 255)
    out[ay0 : ay0 + ah, ax0 : ax0 + aw] = grad
    cv2.rectangle(
        out, (ax0, ay0), (ax0 + aw, ay0 + ah), (150, 150, 150), 1, cv2.LINE_AA
    )
    cv2.putText(out, "0.0", (ax0, ay0 + ah + 12), font, 0.35, color_dim, 1, cv2.LINE_AA)
    cv2.putText(
        out,
        "Activation",
        (ax0 + 40, ay0 + ah + 12),
        font,
        0.35,
        color_dim,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        "1.0",
        (ax0 + aw - 16, ay0 + ah + 12),
        font,
        0.35,
        color_dim,
        1,
        cv2.LINE_AA,
    )


def draw_lift_legend(
    frame_rgb: np.ndarray,
    lift: str,
    time_s: float,
    in_activation_window: bool,
    force_scale_n: float = 500.0,
    force_scale_px: int = 50,
) -> np.ndarray:
    """Draw 2D HUD legend overlay on an RGB video frame.

    Displays:
    1. Title with lift name and current playback timestamp.
    2. Arrow indicators for per-foot GRF contact force (green) and hand grip wrench (blue).
    3. Scaled force reference bar in Newtons (N).
    4. Muscle activation color gradient bar from 0.0 (blue) to 1.0 (red).
    5. OSL-4 activation window status indicator.
    """
    import cv2

    out = frame_rgb.copy()
    h, w = out.shape[:2]
    bx, by, bw, bh = 15, 15, 270, 160
    if bx + bw > w or by + bh > h:
        return out

    sub = out[by : by + bh, bx : bx + bw]
    overlay = np.zeros_like(sub)
    out[by : by + bh, bx : bx + bw] = cv2.addWeighted(overlay, 0.7, sub, 0.3, 0)
    cv2.rectangle(out, (bx, by), (bx + bw, by + bh), (80, 80, 80), 1, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42
    color_text = (230, 230, 230)
    color_dim = (170, 170, 170)

    # 1. Title
    lift_title = lift.replace("_", " ").title()
    title = f"{lift_title} | t = {time_s:.2f} s"
    cv2.putText(
        out, title, (bx + 12, by + 20), font, 0.48, (255, 255, 255), 1, cv2.LINE_AA
    )

    # 2. Arrows & force scale
    _draw_legend_arrows_and_scale(
        out, bx, by, font, scale, color_text, force_scale_n, force_scale_px
    )

    # 3. Activation bar
    _draw_legend_activation_bar(out, bx, by, font, color_dim)

    # 4. Status line
    if in_activation_window:
        status_txt = "activation: OSL-4 active"
        status_col = (100, 255, 100)
    else:
        status_txt = "activation: OSL-4 window only"
        status_col = (200, 200, 120)
    cv2.putText(
        out,
        status_txt,
        (bx + 12, by + bh - 10),
        font,
        0.38,
        status_col,
        1,
        cv2.LINE_AA,
    )

    return out


def _retarget_lift_motion(lift: str, sol: Any) -> Any:
    """Retarget sagittal solution to 3D full body motion."""
    from src.shared.python.lifting.back_bar import BACK_SQUAT, BackBarGeometry
    from src.shared.python.lifting.back_rack import RACKS
    from src.shared.python.lifting.barbell_bushing import Barbell
    from src.shared.python.lifting.bundles import (
        LIFT_SPEC,
        lift_hands_to_full_body,
        lift_to_full_body,
        load_seed_pose,
    )
    from src.shared.python.lifting.loaded_balance import LoadBalance

    lift_spec = LIFT_SPEC.read_bytes()
    if lift == "squat":
        geom = BackBarGeometry.from_standard(
            BACK_SQUAT, height_m=1.78, rack=RACKS["high_bar"]
        )
        load = LoadBalance(Barbell(float(sol.source["bar_mass_kg"])))
        return lift_to_full_body(
            lift_spec, sol, load_seed_pose(), geometry=geom, load=load
        )
    return lift_hands_to_full_body(lift_spec, sol, load_seed_pose(), lift)


def _compute_full_motion_kinematics(
    kin: Any,
    frame: Any,
    attachments: dict[str, Any],
    lift_motion: Any,
    in_pelvis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute landmark positions, barbell endpoints, and pelvis rotations."""
    from src.shared.python.lifting.bundle_targets import IK_LANDMARKS, PELVIS_BODY

    order = tuple(attachments)
    full_points = np.array([kin.marker_positions(q) for q in lift_motion.q])
    full_positions = frame.point(full_points[:, [order.index(x) for x in IK_LANDMARKS]])

    bar_pts = full_positions[:, IK_LANDMARKS.index("bar")]
    ends_1 = bar_pts + np.array([0.0, 0.0, -1.10])
    ends_2 = bar_pts + np.array([0.0, 0.0, 1.10])
    bar_endpoints = np.concatenate([ends_1, ends_2], axis=0)

    full_pelvis_rot = np.array(
        [
            frame.rotation
            @ kin.body_poses(q, [PELVIS_BODY])[PELVIS_BODY][0]
            @ in_pelvis
            for q in lift_motion.q
        ]
    )
    return full_positions, bar_endpoints, full_pelvis_rot


def solve_full_lift_motion_targets(
    lift: str,
    path: Path,
    stride: int = 1,
) -> tuple[LiftTargets, tuple[float, float], np.ndarray]:
    """Extract full-motion LiftTargets from bundle sagittal retargeting provenance.

    Returns (full_targets, (t_start, t_end), bar_endpoints_m).
    """
    from src.shared.python.lifting.bundle_targets import (
        _foot_stations,
        _kinematics,
        _lift_frame,
        bundle_pelvis_rotation,
        lift_targets,
    )
    from src.shared.python.lifting.bundles import (
        LANDMARK_ATTACHMENTS,
        LANDMARK_LABELS,
    )
    from src.shared.python.lifting.sagittal_retarget import SagittalSolution
    from src.shared.python.motion_matching.same_input import InputBundle

    bundle = InputBundle.load(path)
    win_dict = bundle.provenance.get("window", {})
    t_start = float(win_dict.get("start_s", 0.0))
    t_dur = float(win_dict.get("duration_s", 0.2))
    window_s = (t_start, t_start + t_dur)

    targets_short = lift_targets(lift, path, stride=5)
    spec = json.loads(bundle.spec_bytes)
    stations = _foot_stations(spec)
    attachments = {
        label: LANDMARK_ATTACHMENTS[label] for label in LANDMARK_LABELS
    } | stations
    kin, body_mass = _kinematics(bundle.spec_bytes, attachments)
    frame = _lift_frame(kin, bundle.reference_q[0])

    sol = SagittalSolution.from_record(bundle.provenance["sagittal_solution"])
    lift_motion = _retarget_lift_motion(lift, sol)
    in_pelvis = bundle_pelvis_rotation(spec)
    full_positions, bar_endpoints, full_pelvis_rot = _compute_full_motion_kinematics(
        kin, frame, attachments, lift_motion, in_pelvis
    )

    indices = tuple(range(0, len(lift_motion.time_s), stride))
    time_s = lift_motion.time_s[list(indices)]
    full_positions = full_positions[list(indices)]
    full_pelvis_rot = full_pelvis_rot[list(indices)]

    full_targets = LiftTargets(
        lift=lift,
        path=path,
        sha256=targets_short.sha256,
        indices=indices,
        time_s=time_s,
        positions_m=full_positions,
        foot_stations_m=stations,
        grf_force_n=targets_short.grf_force_n,
        grf_moment_nm=targets_short.grf_moment_nm,
        hand_force_n=targets_short.hand_force_n,
        hand_torque_nm=targets_short.hand_torque_nm,
        grip_m=targets_short.grip_m,
        pelvis_rotation=full_pelvis_rot,
        source_coordinates=targets_short.source_coordinates,
        body_mass_kg=body_mass,
        bar_mass_kg=targets_short.bar_mass_kg,
        height_m=targets_short.height_m,
        frame=frame,
        tibial_torsion=targets_short.tibial_torsion,
        bench_force_n=targets_short.bench_force_n,
        bench_point_m=targets_short.bench_point_m,
        provenance=targets_short.provenance,
    )
    return full_targets, window_s, bar_endpoints


def solve_full_lift_motion_and_muscles(
    lift: str,
    stride: int = 5,
    max_ik_frames: int | None = None,
) -> tuple[
    LiftTargets,
    Any,
    Any,
    LiftTargets,
    np.ndarray,
    list[str],
    tuple[float, float],
    np.ndarray,
]:
    """Solve full-motion IK and OSL-4 window static optimization muscle activations.

    Returns (full_targets, build, full_ik, window_targets, activations, muscle_names, window_s, bar_endpoints).
    """
    from src.engines.physics_engines.opensim.python import msk_lift_ik_id as ikid
    from src.engines.physics_engines.opensim.python.msk_lift_arm_reach import (
        reaching_lift_build,
    )
    from src.engines.physics_engines.opensim.python.msk_lift_muscles import (
        MuscleBasis,
        _make_muscles_rigid_tendon,
        _solve_lift_frames,
        leg_coordinates,
    )

    path = SAME_INPUT_LIFT_BUNDLES.get(lift)
    if path is None or not path.is_file():
        raise ValueError(f"no same-input bundle found for lift {lift!r}")

    # 1. Short-window solve for activations & wrenches
    window_targets = lift_targets(lift, path, stride=stride)
    base_model = resolve_base_model()
    scales = ikid.bundle_segment_scales(window_targets, base_model)
    build, _reach = reaching_lift_build(lift, window_targets, base_model, scales)
    pelvis_short = ikid.pelvis_landmark_targets(build, window_targets, base_model)
    ik_short = ikid.run_inverse_kinematics(build, window_targets, pelvis_short)
    inverse_short = ikid.run_inverse_dynamics(build, window_targets, ik_short)

    _make_muscles_rigid_tendon(build.model)
    basis = MuscleBasis(build.model, ik_short.coordinates)
    leg_coords = leg_coordinates()
    leg_indices = [ik_short.coordinates.index(c) for c in leg_coords]
    solve_short = _solve_lift_frames(
        basis, ik_short, inverse_short, leg_indices, window_targets.time_s
    )

    # 2. Full-motion targets & IK
    full_targets, window_s, bar_endpoints = solve_full_lift_motion_targets(
        lift, path, stride=1
    )
    if max_ik_frames is not None and max_ik_frames < len(full_targets.time_s):
        sub_indices = full_targets.indices[:max_ik_frames]
        full_targets = LiftTargets(
            lift=full_targets.lift,
            path=full_targets.path,
            sha256=full_targets.sha256,
            indices=sub_indices,
            time_s=full_targets.time_s[:max_ik_frames],
            positions_m=full_targets.positions_m[:max_ik_frames],
            foot_stations_m=full_targets.foot_stations_m,
            grf_force_n=full_targets.grf_force_n,
            grf_moment_nm=full_targets.grf_moment_nm,
            hand_force_n=full_targets.hand_force_n,
            hand_torque_nm=full_targets.hand_torque_nm,
            grip_m=full_targets.grip_m,
            pelvis_rotation=full_targets.pelvis_rotation[:max_ik_frames],
            source_coordinates=full_targets.source_coordinates,
            body_mass_kg=full_targets.body_mass_kg,
            bar_mass_kg=full_targets.bar_mass_kg,
            height_m=full_targets.height_m,
            frame=full_targets.frame,
            tibial_torsion=full_targets.tibial_torsion,
            bench_force_n=full_targets.bench_force_n,
            bench_point_m=full_targets.bench_point_m,
            provenance=full_targets.provenance,
        )

    pelvis_full = ikid.pelvis_landmark_targets(build, full_targets, base_model)
    full_ik = ikid.run_inverse_kinematics(build, full_targets, pelvis_full)

    return (
        full_targets,
        build,
        full_ik,
        window_targets,
        solve_short.activations,
        basis.muscle_names,
        window_s,
        bar_endpoints,
    )


def solve_lift_motion_and_muscles(
    lift: str,
    stride: int = 5,
) -> tuple[LiftTargets, Any, Any, np.ndarray, list[str]]:
    """Solve IK, ID, and static optimization muscle activations for a barbell lift.

    Returns (targets, build, ik, activations, muscle_names).
    """
    from src.engines.physics_engines.opensim.python import msk_lift_ik_id as ikid
    from src.engines.physics_engines.opensim.python.msk_lift_arm_reach import (
        reaching_lift_build,
    )
    from src.engines.physics_engines.opensim.python.msk_lift_muscles import (
        MuscleBasis,
        _make_muscles_rigid_tendon,
        _solve_lift_frames,
        leg_coordinates,
    )

    path = SAME_INPUT_LIFT_BUNDLES.get(lift)
    if path is None or not path.is_file():
        raise ValueError(f"no same-input bundle found for lift {lift!r}")

    targets = lift_targets(lift, path, stride=stride)
    base_model = resolve_base_model()
    scales = ikid.bundle_segment_scales(targets, base_model)
    build, _reach = reaching_lift_build(lift, targets, base_model, scales)
    pelvis = ikid.pelvis_landmark_targets(build, targets, base_model)
    ik = ikid.run_inverse_kinematics(build, targets, pelvis)
    inverse = ikid.run_inverse_dynamics(build, targets, ik)

    _make_muscles_rigid_tendon(build.model)
    basis = MuscleBasis(build.model, ik.coordinates)
    leg_coords = leg_coordinates()
    leg_indices = [ik.coordinates.index(c) for c in leg_coords]
    solve = _solve_lift_frames(basis, ik, inverse, leg_indices, targets.time_s)

    return targets, build, ik, solve.activations, basis.muscle_names


def generate_contact_sheet(
    frames_or_video: Path | Sequence[np.ndarray],
    out_path: Path,
    count: int = 6,
) -> Path:
    """Generate a 2x3 contact sheet of 6 evenly spaced frames."""
    import imageio.v2 as imageio

    if isinstance(frames_or_video, Path):
        reader = imageio.get_reader(str(frames_or_video))
        all_frames = list(reader)  # type: ignore[call-overload]
        reader.close()
        step_indices = np.linspace(0, len(all_frames) - 1, count, dtype=int)
        frames = [all_frames[i] for i in step_indices]
    else:
        seq = list(frames_or_video)
        step_indices = np.linspace(0, len(seq) - 1, count, dtype=int)
        frames = [seq[i] for i in step_indices]

    row0 = np.concatenate(frames[:3], axis=1)
    row1 = np.concatenate(frames[3:6], axis=1)
    sheet = np.concatenate([row0, row1], axis=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(out_path), sheet)
    return out_path


def render_opensim_lift_video(
    lift: str,
    out_path: Path | str,
    view: str = "side",
    fps: int = 60,
    speed: float = 1.0,
    overlay: str = "forces",
    frames: int | None = None,
    sheet_path: Path | str | None = None,
) -> dict[str, Any]:
    """Render a headless OpenSim barbell lift video."""
    cfg = LiftVideoConfig(lift=lift, view=view, fps=fps, speed=speed, overlay=overlay)
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if cfg.lift == "bench_press":
        return {
            "status": "unavailable",
            "lift": cfg.lift,
            "view": cfg.view,
            "frames": 0,
            "reason": (
                "bench press IK does not pose a supine lifter; "
                "standing posture on bench is unphysical"
            ),
        }

    # Execute render worker via subprocess under xvfb
    worker_script = Path(__file__).resolve().parent / "_msk_lift_render_worker.py"
    with tempfile.TemporaryDirectory(prefix="ud_lift_render_") as tmp:
        work_dir = Path(tmp)
        job_doc = {
            "lift": cfg.lift,
            "view": cfg.view,
            "fps": cfg.fps,
            "speed": cfg.speed,
            "overlay": cfg.overlay,
            "width": cfg.width,
            "height": cfg.height,
            "frames": frames,
            "out_path": str(out_file),
            "sheet_path": str(sheet_path) if sheet_path else None,
        }
        job_file = work_dir / "job.json"
        job_file.write_text(json.dumps(job_doc), encoding="utf-8")

        cmd = [
            "xvfb-run",
            "-a",
            "-s",
            "-screen 0 1280x960x24",
            sys.executable,
            str(worker_script),
            str(job_file),
        ]
        env = {**os.environ}
        if "UPSTREAMDRIFT_MSK_BASE_MODEL" not in env:
            default_base = Path(
                "/home/dieterolson/Repositories/ud-13044/shared/models/opensim/opensim-models/Models/Rajagopal/RajagopalLaiUhlrich2023.osim"
            )
            if default_base.is_file():
                env["UPSTREAMDRIFT_MSK_BASE_MODEL"] = str(default_base)
        env["NATIVE_VIEWER_XVFB"] = "1"
        env["PYTHONPATH"] = os.pathsep.join(sys.path)

        proc = subprocess.run(  # noqa: S603 - fixed argv
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=1800,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Lift render worker failed (code {proc.returncode}):\n"
                f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )

        res_file = work_dir / "result.json"
        if not res_file.is_file():
            raise RuntimeError("Lift render worker produced no result.json")

        result: dict[str, Any] = json.loads(res_file.read_text(encoding="utf-8"))
        return result
