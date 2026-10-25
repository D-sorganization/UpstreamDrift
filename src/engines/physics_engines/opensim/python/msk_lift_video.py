"""Headless OpenSim barbell lift video rendering (OSL-8, #13045).

Public entry points re-export framing and motion helpers and spawn the xvfb
render worker for MP4 export and contact sheets.
"""

from __future__ import annotations

from collections.abc import Sequence
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np

from src.engines.physics_engines.opensim.python.msk_lift_video_framing import (
    BARBELL_LIFTS,
    LIFT_OVERLAYS,
    LIFT_VIEWS,
    SETTLE_S,
    UI_STRIP_PX,
    VIEWER_FOV_Y_RAD,
    LiftVideoConfig,
    compute_lift_framing,
    draw_lift_legend,
    extract_lift_wrenches,
    lift_camera_transforms,
    lift_video_schedule,
    muscle_color_rgb,
    pinhole_for_lifting_view,
)
from src.engines.physics_engines.opensim.python.msk_lift_video_motion import (
    solve_full_lift_motion_and_muscles,
    solve_full_lift_motion_targets,
    solve_lift_motion_and_muscles,
)
from src.engines.physics_engines.opensim.python.musculoskeletal_swing import (
    resolve_base_model,
)

__all__ = [
    "BARBELL_LIFTS",
    "LIFT_OVERLAYS",
    "LIFT_VIEWS",
    "SETTLE_S",
    "UI_STRIP_PX",
    "VIEWER_FOV_Y_RAD",
    "LiftVideoConfig",
    "compute_lift_framing",
    "draw_lift_legend",
    "extract_lift_wrenches",
    "generate_contact_sheet",
    "lift_camera_transforms",
    "lift_video_schedule",
    "muscle_color_rgb",
    "pinhole_for_lifting_view",
    "render_opensim_lift_video",
    "solve_full_lift_motion_and_muscles",
    "solve_full_lift_motion_targets",
    "solve_lift_motion_and_muscles",
]


def generate_contact_sheet(
    frames_or_video: Path | Sequence[np.ndarray],
    out_path: Path,
    count: int = 6,
    window_frame_indices: Sequence[int] | None = None,
) -> Path:
    """Generate a 2x3 contact sheet of 6 representative frames.

    When ``window_frame_indices`` is provided, ensures at least 2 frames capture
    the active OSL-4 window with visible force overlays and muscle activations.
    """
    import imageio.v2 as imageio

    if isinstance(frames_or_video, Path):
        reader = imageio.get_reader(str(frames_or_video))
        all_frames = list(reader)  # type: ignore[call-overload]
        reader.close()
    else:
        all_frames = list(frames_or_video)

    total = len(all_frames)
    if (
        window_frame_indices is not None
        and len(window_frame_indices) > 0
        and count == 6
        and total >= 6
    ):
        w_start = int(window_frame_indices[0])
        w_end = int(window_frame_indices[-1])
        w_mid1 = w_start + (w_end - w_start) // 3
        w_mid2 = w_start + 2 * (w_end - w_start) // 3
        pre1 = 0
        pre2 = w_start // 2
        post1 = w_end + (total - 1 - w_end) // 2
        post2 = total - 1
        step_indices = np.array([pre1, pre2, w_mid1, w_mid2, post1, post2], dtype=int)
    else:
        step_indices = np.linspace(0, total - 1, count, dtype=int)

    frames = [all_frames[i] for i in step_indices]

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
            try:
                env["UPSTREAMDRIFT_MSK_BASE_MODEL"] = str(resolve_base_model())
            except FileNotFoundError:
                pass
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
