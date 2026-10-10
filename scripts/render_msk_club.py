#!/usr/bin/env python3
"""Render the Rajagopal golf humanoid swinging the shared two-hand club (OSV-9).

The generated full-body driver swing (``tests/fixtures/club_face/
swing_q_driver.npz``, 2 ms frames) is tracked by the golf humanoid with
:func:`msk_club_tracking.track_swing`, then drawn by the simbody visualizer
on a virtual X display: address, top, impact and finish stills from the
face-on and down-the-line presets, and a half-speed face-on clip.

Run only under xvfb (never the real display)::

    NATIVE_VIEWER_XVFB=1 xvfb-run -a -s "-screen 0 1280x960x24" \\
        python3 scripts/render_msk_club.py --geometry <opensim Geometry dir>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.engines.physics_engines.opensim.python import club_visuals  # noqa: E402
from src.engines.physics_engines.opensim.python import msk_club as mc  # noqa: E402
from src.engines.physics_engines.opensim.python import (  # noqa: E402
    msk_club_calibration as cal,
)
from src.engines.physics_engines.opensim.python import (  # noqa: E402
    msk_club_tracking as mt,
)
from src.shared.python.golf_view_presets import simbody_camera_transform  # noqa: E402
from src.tools.native_viewer_export.backends._scene import fit_to_size  # noqa: E402
from src.tools.native_viewer_export.backends.opensim_worker import (  # noqa: E402
    FOV_Y_RAD,
    SETTLE_S,
    UI_STRIP_PX,
    grab_window,
    require_virtual_display,
)

logger = logging.getLogger(__name__)

SWING_DT_S = 0.002
#: Generated-model event times (``tests/fixtures/club_face/provenance.json``).
EVENTS_S = {"address": 0.0, "top": 1.104, "impact": 1.326}
CLIP_FPS = 30
SPEED = 0.5
VIEWS = ("face_on", "down_the_line")
DISTANCE_M = 4.2
SIZE = (960, 720)
MAX_DRAWS = 12
DEFAULT_OUT = Path.home() / "Videos" / "Parity Audit" / "golfer_realism" / "msk_club"


def event_frames(n_frames: int) -> dict[str, int]:
    """Fixture frame index of each still (finish is the last frame)."""
    frames = {k: int(round(t / SWING_DT_S)) for k, t in EVENTS_S.items()}
    frames["finish"] = n_frames - 1
    return frames


def clip_frames(n_frames: int) -> list[int]:
    """Fixture frames sampled so ``CLIP_FPS`` playback runs at ``SPEED``."""
    step_s = SPEED / CLIP_FPS
    count = int((n_frames - 1) * SWING_DT_S / step_s) + 1
    return [int(round(i * step_s / SWING_DT_S)) for i in range(count)]


class _Viewer:
    """Simbody visualizer of the golf humanoid on the virtual display."""

    def __init__(self, osim: Any, model_path: Path, geometry: Path | None) -> None:
        self.osim = osim
        club_visuals.register_geometry_path()
        if geometry is not None:
            osim.ModelVisualizer.addDirToGeometrySearchPaths(str(geometry))
        self.model = osim.Model(str(model_path))
        self.model.setUseVisualizer(True)
        self.state = self.model.initSystem()
        viz = self.model.updVisualizer().updSimbodyVisualizer()
        viz.setBackgroundType(viz.GroundAndSky)
        viz.setShowFrameRate(False)
        viz.setShowSimTime(False)
        viz.setCameraFieldOfView(FOV_Y_RAD)
        self.viz = viz
        self.coords = self.model.getCoordinateSet()
        time.sleep(3.0)

    def pose(self, q: dict[str, float]) -> None:
        for name, value in q.items():
            self.coords.get(name).setValue(self.state, float(value), False)
        self.model.realizePosition(self.state)

    def camera(self, view: str, lookat_os: np.ndarray, floor_z: float) -> None:
        lookat = cal.NATIVE_TO_OPENSIM.T @ lookat_os + np.array([0.0, 0.0, floor_z])
        rows, pos = simbody_camera_transform(view, lookat, DISTANCE_M)
        rot = cal.NATIVE_TO_OPENSIM @ np.array(rows)
        pos_os = cal.NATIVE_TO_OPENSIM @ (np.array(pos) - np.array([0, 0, floor_z]))
        mat = self.osim.Mat33()
        for i in range(3):
            for j in range(3):
                mat.set(i, j, float(rot[i, j]))
        transform = self.osim.Transform(
            self.osim.Rotation(mat), self.osim.Vec3(*map(float, pos_os))
        )
        self.viz.setCameraTransform(transform)

    def grab(self) -> np.ndarray:
        """Window image once it has settled on the current pose and camera.

        The visualizer window trails the draw calls by a queue of frames, so
        redraw until three consecutive grabs agree (at most ``MAX_DRAWS``).
        """
        frames: list[np.ndarray] = []
        for _ in range(MAX_DRAWS):
            self.viz.drawFrameNow(self.state)
            time.sleep(SETTLE_S)
            frames.append(grab_window())
            if len(frames) >= 3 and all(
                np.array_equal(frames[-1], f) for f in frames[-3:-1]
            ):
                break
        else:
            logger.warning("visualizer did not settle in %d draws", MAX_DRAWS)
        return fit_to_size(frames[-1][:-UI_STRIP_PX], *SIZE)

    def face_deg(self) -> float:
        return cal.face_angle_deg(self.model, self.state, mc.load_msk_club())


def _write_png(path: Path, frame: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(frame).save(path)


def render(model_path: Path, out: Path, geometry: Path | None) -> dict[str, Any]:
    """Track the swing, write stills, clip and a JSON summary to ``out``."""
    require_virtual_display()
    import imageio.v2 as imageio
    import opensim as osim

    rows = np.load(mc.REPO_ROOT / "tests/fixtures/club_face/swing_q_driver.npz")["q"]
    events, clip = event_frames(len(rows)), clip_frames(len(rows))
    wanted = sorted(set(clip) | set(events.values()))
    tracked = dict(zip(wanted, mt.track_swing(model_path, rows[wanted]), strict=True))
    out.mkdir(parents=True, exist_ok=True)
    viewer = _Viewer(osim, model_path, geometry)
    floor = cal.GeneratedSwing("driver").floor_native_z
    viewer.pose(tracked[0].q)
    pelvis = viewer.model.getBodySet().get("pelvis")
    lookat = cal.transform_in_ground(pelvis, viewer.state)[:3, 3] * [1.0, 0.0, 1.0]
    lookat[1] = 0.95
    summary: dict[str, Any] = {"model": model_path.stem, "events": {}}
    for event, k in events.items():
        viewer.pose(tracked[k].q)
        summary["events"][event] = {
            "frame": k,
            "time_s": k * SWING_DT_S,
            "face_deg": viewer.face_deg(),
            "lead_grip_error_m": tracked[k].lead_grip_error_m,
            "trail_grip_gap_m": tracked[k].trail_grip_gap_m,
            "landmark_rms_m": tracked[k].landmark_rms_m,
        }
        for view in VIEWS:
            viewer.camera(view, lookat, floor)
            _write_png(out / f"msk_club_{event}_{view}.png", viewer.grab())
    clip_path = out / "msk_club_swing_half_speed_face_on.mp4"
    writer = imageio.get_writer(str(clip_path), fps=CLIP_FPS, codec="libx264")
    viewer.camera("face_on", lookat, floor)
    for k in clip:
        viewer.pose(tracked[k].q)
        writer.append_data(viewer.grab())
    writer.close()
    summary["clip"] = {"path": clip_path.name, "frames": len(clip), "speed": SPEED}
    summary["max_lead_grip_error_m"] = max(
        f.lead_grip_error_m for f in tracked.values()
    )
    summary["max_trail_grip_gap_m"] = max(f.trail_grip_gap_m for f in tracked.values())
    (out / "msk_club_render_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model", type=Path, default=mc.MODELS_DIR / "golf_humanoid.osim"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--geometry", type=Path, default=None, help="OpenSim Geometry")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    summary = render(args.model, args.out, args.geometry)
    logger.info("%s", json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
