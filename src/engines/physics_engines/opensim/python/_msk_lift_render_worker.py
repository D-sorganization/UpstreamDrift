"""OpenSim lift video render worker running under virtual X display (OSL-8, #13045).

Subprocess invoked by ``render_opensim_lift_video`` under ``xvfb-run``.
Captures the Simbody visualizer frame-by-frame with dynamic muscle recoloring
and projected force/wrench overlays.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time
from typing import Any, NamedTuple

import imageio.v2 as imageio
import numpy as np

from src.engines.physics_engines.opensim.python.msk_lift_video import (
    SETTLE_S,
    UI_STRIP_PX,
    VIEWER_FOV_Y_RAD,
    compute_lift_framing,
    draw_lift_legend,
    extract_lift_wrenches,
    generate_contact_sheet,
    lift_camera_transforms,
    lift_video_schedule,
    muscle_color_rgb,
    pinhole_for_lifting_view,
    solve_full_lift_motion_and_muscles,
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


def _setup_visualizer(osim: Any, model: Any) -> Any:
    """Initialize Simbody visualizer in Y-up orientation with neutral ground."""
    from src.engines.physics_engines.opensim.python.musculoskeletal_swing import (
        resolve_base_model,
    )

    try:
        base_path = resolve_base_model()
        # Add Rajagopal-specific Geometry (contains r_pelvis.vtp, l_pelvis.vtp, sacrum.vtp)
        # as well as the root Geometry directory (OSL-8, #13045).
        search_dirs = [
            base_path.parent / "Geometry",
            base_path.parents[2] / "Geometry",
        ]
        for d in search_dirs:
            if d.is_dir():
                osim.ModelVisualizer.addDirToGeometrySearchPaths(str(d))
    except (FileNotFoundError, IndexError):
        pass

    # Clear base model experimental markers (C7, CLAV, RACR, etc.) and IK markers
    # so they do not render as floating dark red spheres.
    model.updMarkerSet().clearAndDestroy()

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


def _set_neutral_muscles(osim: Any, model: Any, state: Any) -> None:
    """Set all muscle paths to neutral grey (0.6, 0.6, 0.6)."""
    muscles = model.getMuscles()
    n_muscles = muscles.getSize()
    grey = osim.Vec3(0.6, 0.6, 0.6)
    for i in range(n_muscles):
        model.getMuscles().get(i).updGeometryPath().setColor(state, grey)


FRAME_SETTLE_S: float = 0.05


class ActivationWindow(NamedTuple):
    targets: Any
    activations: np.ndarray
    window_s: tuple[float, float]


def _setup_camera_and_viewport(
    viz: Any,
    state: Any,
    full_targets: Any,
    bar_endpoints_m: np.ndarray,
    job: dict[str, Any],
) -> tuple[tuple[int, int, int, int], float, tuple[float, float, float], float, int]:
    """Configure visualizer camera, viewport crop, and physical force legend scale."""
    osim = import_opensim()
    viz.drawFrameNow(state)
    time.sleep(SETTLE_S)
    win = grab_window()
    win_hw = win.shape[:2]
    crop_box = principal_crop(win_hw, UI_STRIP_PX, (job["width"], job["height"]))
    y0, y1, _, _ = crop_box
    fov = simbody_fov_y_rad(VIEWER_FOV_Y_RAD, win_hw[0], y1 - y0)
    viz.setCameraFieldOfView(fov)

    lookat, dist = compute_lift_framing(
        full_targets.positions_m,
        bar_endpoints_m,
        view=job["view"],
        fov_y_rad=fov,
        image_size_px=(job["width"], job["height"]),
    )
    _set_simbody_camera(osim, viz, job["view"], lookat, dist)

    focal = (job["height"] / 2.0) / math.tan(fov / 2.0)
    scale_px = int(round(500.0 * 0.001 * focal / dist))
    scale_px = max(20, min(120, scale_px))
    return crop_box, fov, lookat, dist, scale_px


def _render_frames(
    job: dict[str, Any],
    full_targets: Any,
    model: Any,
    state: Any,
    viz: Any,
    full_ik: Any,
    act_window: ActivationWindow,
    bar_endpoints_m: np.ndarray,
) -> tuple[list[np.ndarray], list[int]]:
    """Render and capture all requested video frames with overlays and dynamic framing."""
    osim = import_opensim()
    window_targets = act_window.targets
    activations = act_window.activations
    window_s = act_window.window_s

    sched, q_interp, _ = lift_video_schedule(
        full_targets.time_s, full_ik.q, fps=job["fps"], speed=job["speed"]
    )
    total_frames = len(sched.sample_times_s)
    if job.get("frames") is not None:
        total_frames = min(int(job["frames"]), total_frames)

    (y0, y1, x0, x1), fov, lookat, dist, scale_px = _setup_camera_and_viewport(
        viz, state, full_targets, bar_endpoints_m, job
    )

    t_start, t_end = window_s
    coords = model.getCoordinateSet()
    style = ForceGlyphStyle(
        force_scale_m_per_n=1.0 / 1000.0,
        max_length_m=1.0,
        shaft_radius_m=0.015,
    )
    rendered: list[np.ndarray] = []
    window_frame_indices: list[int] = []

    for f_idx in range(total_frames):
        t_curr = float(sched.sample_times_s[f_idx])
        for name, val in zip(full_ik.coordinates, q_interp[f_idx], strict=True):
            coords.get(name).setValue(state, float(val), False)

        in_window = (t_curr >= t_start - 1e-4) and (t_curr <= t_end + 1e-4)
        if in_window:
            window_frame_indices.append(f_idx)
            t_rel = float(np.clip(t_curr - t_start, 0.0, window_targets.time_s[-1]))
            window_idx = int(np.argmin(np.abs(window_targets.time_s - t_rel)))
            _recolor_muscles(osim, model, state, activations[window_idx])
        else:
            _set_neutral_muscles(osim, model, state)

        model.realizePosition(state)
        viz.drawFrameNow(state)
        time.sleep(FRAME_SETTLE_S)

        frame = grab_window()
        if job["overlay"] == "forces" and in_window:
            wrenches = extract_lift_wrenches(window_targets, window_idx)
            ft = ForceTorqueFrame(
                time_s=t_curr,
                wrenches=tuple(wrenches),
                engine="opensim",
            )
            cam = pinhole_for_lifting_view(
                job["view"],
                lookat,
                dist,
                fov,
                (frame.shape[1], frame.shape[0]),
            )
            frame = draw_glyphs_rgb(frame, build_glyphs(ft, style), cam)

        cropped = frame[y0:y1, x0:x1]
        final_frame = fit_to_size(cropped, job["width"], job["height"])
        final_frame = draw_lift_legend(
            final_frame,
            lift=job["lift"],
            time_s=t_curr,
            in_activation_window=in_window,
            force_scale_n=500.0,
            force_scale_px=scale_px,
        )
        rendered.append(final_frame)

    return rendered, window_frame_indices


def run_worker(job_path: Path) -> None:
    """Entry point for lift rendering worker."""
    require_virtual_display()
    job: dict[str, Any] = json.loads(job_path.read_text(encoding="utf-8"))

    osim = import_opensim()
    max_ik = int(job["frames"]) if job.get("frames") is not None else None
    (
        full_targets,
        build,
        full_ik,
        window_targets,
        activations,
        _,
        window_s,
        bar_endpoints,
    ) = solve_full_lift_motion_and_muscles(job["lift"], stride=5, max_ik_frames=max_ik)
    state, viz = _setup_visualizer(osim, build.model)

    act_window = ActivationWindow(
        targets=window_targets,
        activations=activations,
        window_s=window_s,
    )
    frames, window_frames = _render_frames(
        job,
        full_targets,
        build.model,
        state,
        viz,
        full_ik,
        act_window,
        bar_endpoints,
    )

    out_file = Path(job["out_path"])
    out_file.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(out_file),
        format="FFMPEG",  # type: ignore[arg-type]
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
        generate_contact_sheet(
            frames, sheet_path, count=6, window_frame_indices=window_frames
        )

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
