"""OpenSim lift video render worker running under virtual X display (OSL-8, #13045).

Subprocess invoked by ``render_opensim_lift_video`` under ``xvfb-run``.
Captures the Simbody visualizer frame-by-frame with dynamic muscle recoloring
and projected force/wrench overlays.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any

import imageio.v2 as imageio
import numpy as np

from src.engines.physics_engines.opensim.python.msk_lift_video import (
    SETTLE_S,
    UI_STRIP_PX,
    VIEWER_FOV_Y_RAD,
    extract_lift_wrenches,
    generate_contact_sheet,
    lift_camera_transforms,
    lift_video_schedule,
    muscle_color_rgb,
    pinhole_for_lifting_view,
    solve_lift_motion_and_muscles,
)
from src.shared.python.engine_core.opensim_runtime import import_opensim
from src.shared.python.force_overlay.contracts import ForceTorqueFrame
from src.shared.python.force_overlay.glyphs import ForceGlyphStyle, build_glyphs
from src.shared.python.logging_pkg.logging_config import get_logger
from src.tools.native_viewer_export.backends._opensim_framing import (
    principal_crop,
    simbody_fov_y_rad,
)
from src.tools.native_viewer_export.backends._scene import fit_to_size
from src.tools.native_viewer_export.backends.opensim_worker import (
    grab_window,
    require_virtual_display,
)
from src.tools.native_viewer_export.overlay2d import draw_glyphs_rgb

logger = get_logger(__name__)

LOOKAT_CENTER_M = (0.0, 0.9, 0.0)
DEFAULT_LIFT_CAMERA_DISTANCE_M = 2.8


def _setup_visualizer(osim: Any, model: Any) -> Any:
    """Initialize Simbody visualizer in Y-up orientation with neutral ground."""
    geom_dir = Path(
        "/home/dieterolson/Repositories/ud-13044/shared/models/opensim/opensim-models/Geometry"
    )
    if geom_dir.is_dir():
        osim.ModelVisualizer.addDirToGeometrySearchPaths(str(geom_dir))

    model.setUseVisualizer(True)
    state = model.initSystem()
    viz = model.updVisualizer().updSimbodyVisualizer()
    viz.setBackgroundType(viz.GroundAndSky)
    viz.setShowFrameRate(False)
    viz.setShowSimTime(False)
    viz.setSystemUpDirection(osim.CoordinateDirection(osim.CoordinateAxis(1)))
    viz.setGroundHeight(0.0)
    return state, viz


def _set_simbody_camera(
    osim: Any,
    viz: Any,
    view: str,
    lookat: tuple[float, float, float],
    distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Point the Simbody visualizer camera according to view preset."""
    simbody_rot, pinhole_rot, cam_pos = lift_camera_transforms(view, lookat, distance)
    mat = osim.Mat33()
    for i in range(3):
        for j in range(3):
            mat.set(i, j, float(simbody_rot[i, j]))
    viz.setCameraTransform(osim.Transform(osim.Rotation(mat), osim.Vec3(*cam_pos)))
    return pinhole_rot, cam_pos


def _recolor_muscles(
    osim: Any,
    model: Any,
    state: Any,
    activations: np.ndarray,
) -> None:
    """Recolor model muscles from blue (0.0) to red (1.0) for the current state."""
    muscles = model.getMuscles()
    n_muscles = muscles.getSize()
    for i in range(n_muscles):
        m = muscles.get(i)
        r, g, b = muscle_color_rgb(float(activations[i]))
        m.updGeometryPath().setColor(state, osim.Vec3(r, g, b))


def _render_frames(
    job: dict[str, Any],
    targets: Any,
    model: Any,
    state: Any,
    viz: Any,
    ik: Any,
    activations: np.ndarray,
) -> list[np.ndarray]:
    """Render and capture all requested video frames with overlays."""
    osim = import_opensim()
    sched, q_interp, nearest = lift_video_schedule(
        targets.time_s, ik.q, fps=job["fps"], speed=job["speed"]
    )
    total_frames = len(sched.sample_times_s)
    if job.get("frames") is not None:
        total_frames = min(int(job["frames"]), total_frames)

    viz.drawFrameNow(state)
    time.sleep(SETTLE_S)
    win = grab_window()
    win_hw = win.shape[:2]
    y0, y1, x0, x1 = principal_crop(win_hw, UI_STRIP_PX, (job["width"], job["height"]))
    fov = simbody_fov_y_rad(VIEWER_FOV_Y_RAD, win_hw[0], y1 - y0)
    viz.setCameraFieldOfView(fov)

    dist = DEFAULT_LIFT_CAMERA_DISTANCE_M
    _set_simbody_camera(osim, viz, job["view"], LOOKAT_CENTER_M, dist)

    coords = model.getCoordinateSet()
    style = ForceGlyphStyle()
    rendered: list[np.ndarray] = []

    for f_idx in range(total_frames):
        for name, val in zip(ik.coordinates, q_interp[f_idx], strict=True):
            coords.get(name).setValue(state, float(val), False)

        orig_idx = int(nearest[f_idx])
        _recolor_muscles(osim, model, state, activations[orig_idx])
        model.realizePosition(state)
        viz.drawFrameNow(state)
        time.sleep(SETTLE_S)

        frame = grab_window()
        if job["overlay"] == "forces":
            wrenches = extract_lift_wrenches(targets, orig_idx)
            ft = ForceTorqueFrame(
                time_s=float(sched.sample_times_s[f_idx]),
                wrenches=tuple(wrenches),
                engine="opensim",
            )
            cam = pinhole_for_lifting_view(
                job["view"],
                LOOKAT_CENTER_M,
                dist,
                fov,
                (frame.shape[1], frame.shape[0]),
            )
            frame = draw_glyphs_rgb(frame, build_glyphs(ft, style), cam)

        cropped = frame[y0:y1, x0:x1]
        final_frame = fit_to_size(cropped, job["width"], job["height"])
        rendered.append(final_frame)

    return rendered


def run_worker(job_path: Path) -> None:
    """Entry point for lift rendering worker."""
    require_virtual_display()
    job: dict[str, Any] = json.loads(job_path.read_text(encoding="utf-8"))

    osim = import_opensim()
    targets, build, ik, activations, _ = solve_lift_motion_and_muscles(
        job["lift"], stride=5
    )
    state, viz = _setup_visualizer(osim, build.model)

    frames = _render_frames(job, targets, build.model, state, viz, ik, activations)

    out_file = Path(job["out_path"])
    out_file.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(out_file),
        format="FFMPEG",
        fps=int(job["fps"]),
        codec="libx264",
        output_params=["-crf", "18", "-pix_fmt", "yuv420p"],
        macro_block_size=None,
    )
    for f in frames:
        writer.append_data(f)
    writer.close()

    if job.get("sheet_path"):
        sheet_path = Path(job["sheet_path"])
        generate_contact_sheet(frames, sheet_path, count=6)

    result = {
        "status": "rendered",
        "lift": job["lift"],
        "view": job["view"],
        "frames": len(frames),
        "fps": job["fps"],
        "speed": job["speed"],
        "overlay": job["overlay"],
        "out_path": str(out_file),
        "sheet_path": job.get("sheet_path"),
    }
    result_path = job_path.parent / "result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run_worker(Path(sys.argv[1]))
