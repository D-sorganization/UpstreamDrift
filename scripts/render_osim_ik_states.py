"""Render an OpenSim model driven by an IK states file (MATCH-3 #14445).

The OpenSim half of render_best_matches.py: xvfb only, Simbody visualizer frames
cropped with the native-viewer helpers; 0.5x and 1x MP4s per view plus stills.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time

import imageio.v3 as iio
import imageio_ffmpeg
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.tools.native_viewer_export.backends._opensim_framing import (  # noqa: E402
    principal_crop,
    simbody_fov_y_rad,
)
from src.tools.native_viewer_export.backends._scene import fit_to_size  # noqa: E402
from src.tools.native_viewer_export.backends.opensim_worker import (  # noqa: E402
    UI_STRIP_PX,
    grab_window,
    init_simbody_visualizer,
    require_virtual_display,
)

WIDTH, HEIGHT = 1920, 1080
FOV_Y_RAD = 0.7
SETTLE_S = 0.3
VIEWS = ("face_on", "down_the_line")
#: camera direction per view in the Y-up OpenSim ground frame
VIEW_DIRECTIONS = {"face_on": (1.0, 0.0, 0.0), "down_the_line": (0.0, 0.0, 1.0)}
STILL_FRACTIONS = {"address": 0.0, "top": 0.7505, "impact": 0.8998, "finish": 1.0}
HEIGHT_MARGIN = 1.18
#: single-threaded x264 so identical frames encode to identical files
ENCODER = {"codec": "libx264", "quality": 8, "output_params": ["-threads", "1"]}


def load_states(osim, path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    """``(time, coordinate names, q rows)`` from an OpenSim states table."""
    table = osim.TimeSeriesTable(str(path))
    names = [label.split("/")[-2] for label in table.getColumnLabels()]
    times = np.asarray(table.getIndependentColumn(), dtype=float)
    q = np.asarray(table.getMatrix().to_numpy(), dtype=float)
    if q.shape != (times.size, len(names)) or np.abs(np.diff(q, axis=0)).max() > 1:
        raise ValueError(f"{path.name}: states are not a continuous trajectory")
    return times, names, q


class OsimRenderer:
    """Poses the model at a time, points the camera and grabs a cropped frame."""

    def __init__(self, osim, model_path: Path, states_path: Path) -> None:
        self.osim = osim
        self.model = osim.Model(str(model_path))
        self.state, self.viz = init_simbody_visualizer(osim, self.model, up_axis=1)
        self.times, self.names, self.q = load_states(osim, states_path)
        self.coords = self.model.getCoordinateSet()
        time.sleep(3.0)
        self.viz.drawFrameNow(self.state)
        time.sleep(SETTLE_S)
        window_hw = grab_window().shape[:2]
        self.crop = principal_crop(window_hw, UI_STRIP_PX, (WIDTH, HEIGHT))
        crop_h = self.crop[1] - self.crop[0]
        self.viz.setCameraFieldOfView(
            simbody_fov_y_rad(FOV_Y_RAD, window_hw[0], crop_h)
        )
        self.look, self.distance = self._framing()

    def pose(self, t: float) -> None:
        for j, name in enumerate(self.names):
            value = float(np.interp(t, self.times, self.q[:, j]))
            self.coords.get(name).setValue(self.state, value, False)
        self.model.realizePosition(self.state)

    def _framing(self) -> tuple[np.ndarray, float]:
        bodies = self.model.getBodySet()
        points = []
        for t in self.times[::10]:
            self.pose(float(t))
            for b in range(bodies.getSize()):
                p = bodies.get(b).getPositionInGround(self.state)
                points.append([p.get(0), p.get(1), p.get(2)])
        lo, hi = np.min(points, axis=0), np.max(points, axis=0)
        look = (lo + hi) / 2
        look[1] = (max(lo[1], 0.0) + hi[1]) / 2
        view_h = hi[1] * HEIGHT_MARGIN + 0.1
        return look, view_h / 2 / math.tan(FOV_Y_RAD / 2)

    def _camera(self, view: str) -> None:
        forward = np.asarray(VIEW_DIRECTIONS[view])
        up = np.array([0.0, 1.0, 0.0])
        rot = np.column_stack([np.cross(forward, up), up, -forward])
        mat = self.osim.Mat33()
        for i in range(3):
            for j in range(3):
                mat.set(i, j, float(rot[i, j]))
        pos = self.osim.Vec3(*map(float, self.look - forward * self.distance))
        self.viz.setCameraTransform(self.osim.Transform(self.osim.Rotation(mat), pos))

    def frame(self, view: str, t: float) -> np.ndarray:
        self.pose(t)
        self._camera(view)
        self.viz.drawFrameNow(self.state)
        time.sleep(SETTLE_S)
        y0, y1, x0, x1 = self.crop
        return fit_to_size(grab_window()[y0:y1, x0:x1], WIDTH, HEIGHT)


def _writer(path: Path, fps: int):
    gen = imageio_ffmpeg.write_frames(str(path), (WIDTH, HEIGHT), fps=fps, **ENCODER)
    gen.send(None)
    return gen


def render(renderer: OsimRenderer, out: Path, fps: int, limit: int) -> int:
    """Write stills and videos; return the number of 0.5x frames per view."""
    end = float(renderer.times[-1])
    for name, fraction in STILL_FRACTIONS.items():
        iio.imwrite(
            out / f"still_face_on_{name}.png", renderer.frame("face_on", end * fraction)
        )
    step = 0.5 / fps
    count = int(end / step) + 1
    count = min(count, limit) if limit else count
    writers = {
        (v, s): _writer(out / f"os9_{v}_{s}_{fps}fps.mp4", fps)
        for v in VIEWS
        for s in ("0p5x", "1x")
    }
    for k in range(count):
        for view in VIEWS:
            img = renderer.frame(view, k * step)
            writers[(view, "0p5x")].send(img)
            if k % 2 == 0:
                writers[(view, "1x")].send(img)
    for gen in writers.values():
        gen.close()
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("--model", "--states", "--geometry", "--out"):
        parser.add_argument(flag, type=Path, required=True)
    parser.add_argument("--fps", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    require_virtual_display()
    import opensim as osim

    osim.Logger.setLevelString("Error")
    osim.ModelVisualizer.addDirToGeometrySearchPaths(str(args.geometry))
    args.out.mkdir(parents=True, exist_ok=True)
    renderer = OsimRenderer(osim, args.model, args.states)
    count = render(renderer, args.out, args.fps, args.limit)
    print(f"rendered {count} frames per view at 0.5x")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
