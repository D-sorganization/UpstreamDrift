"""Render the 4 barbell lift showcase videos and contact sheets (OSL-8, #13045)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import os
import tempfile
import time
from typing import Any

from src.engines.physics_engines.opensim.python.msk_lift_video import (
    generate_contact_sheet,
    render_opensim_lift_video,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger("render_osl8_videos")

_SCRATCHPAD_NAME = "claude-1000/-home-dieterolson-Repositories-UpstreamDrift/e275def1-d52e-475b-9174-6844c12cb3cc/scratchpad/agy/osl8-videos"
OUT_DIR = Path(
    os.environ.get(
        "OSL8_VIDEOS_DIR",
        str(Path(tempfile.gettempdir()) / _SCRATCHPAD_NAME),
    )
)
LIFTS = ("squat", "deadlift", "snatch", "clean_and_jerk")


def render_all() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"rendered": {}, "bench_press": {}}

    # Bench press record
    summary["bench_press"] = {
        "status": "unavailable",
        "reason": (
            "bench press IK does not pose a supine lifter; "
            "standing posture on bench is unphysical"
        ),
    }
    logger.info(
        "Recorded bench_press as unavailable: %s", summary["bench_press"]["reason"]
    )

    for lift in LIFTS:
        logger.info("Starting render for lift: %s", lift)
        t0 = time.time()
        video_path = OUT_DIR / f"{lift}_side_60fps_forces.mp4"
        sheet_path = OUT_DIR / f"{lift}_contact_sheet.png"

        result = render_opensim_lift_video(
            lift=lift,
            out_path=video_path,
            view="side",
            fps=60,
            speed=1.0,
            overlay="forces",
            sheet_path=sheet_path,
        )
        elapsed = time.time() - t0

        logger.info(
            "Video and contact sheet rendered in %.2fs: %s (%d frames)",
            elapsed,
            video_path,
            result.get("frames", 0),
        )

        vid_size = video_path.stat().st_size
        sheet_size = sheet_path.stat().st_size if sheet_path.is_file() else 0

        summary["rendered"][lift] = {
            "lift": lift,
            "view": "side",
            "fps": 60,
            "speed": 1.0,
            "overlay": "forces",
            "video_path": str(video_path),
            "video_bytes": vid_size,
            "frames": result.get("frames", 0),
            "duration_s": result.get("frames", 0) / 60.0,
            "render_time_s": elapsed,
            "contact_sheet_path": str(sheet_path),
            "contact_sheet_bytes": sheet_size,
        }

    return summary


if __name__ == "__main__":
    res = render_all()
    print(json.dumps(res, indent=2))
