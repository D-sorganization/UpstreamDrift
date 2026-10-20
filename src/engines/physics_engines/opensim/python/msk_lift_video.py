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
