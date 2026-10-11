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

With ``--capture`` (a private C3D; its path is never written to the outputs,
only the boolean ``capture_used``) the stills use the capture's address, top
and impact times and the tracking follows ``--mode`` (``baseline``, ``turn``
pelvis and upper-trunk turn targets, or ``turn_feet`` plus capture-planted
feet; default ``turn_feet``) for ``--club`` ``driver`` or ``iron7``::

    NATIVE_VIEWER_XVFB=1 xvfb-run -a -s "-screen 0 1280x960x24" \\
        python3 scripts/render_msk_club.py --geometry <opensim Geometry dir> \\
        --capture "$CAPTURE_DATA_DIR/<capture>.c3d" --club driver --mode turn_feet

Output goes to ``<default dir>/<club>_<mode>`` unless ``--out`` is given.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, NamedTuple

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
from src.engines.physics_engines.opensim.python import (  # noqa: E402
    msk_turn_targets as tt,
)
from src.shared.python.golf_view_presets import simbody_camera_transform  # noqa: E402
from src.shared.python.motion_matching.turn_receipt import (  # noqa: E402
    capture_turn_inputs,
)
from src.shared.python.swing_comparison.turn import (  # noqa: E402
    model_turn_lines,
    turn_source_block,
)
from src.tools.native_viewer_export.backends._scene import fit_to_size  # noqa: E402
from src.tools.native_viewer_export.backends.opensim_worker import (  # noqa: E402
    FOV_Y_RAD,
    SETTLE_S,
    UI_STRIP_PX,
    grab_window,
    require_virtual_display,
)

logger = logging.getLogger(__name__)

SWING_DT_S = mt.FIXTURE_DT_S
#: Generated-model event times (``tests/fixtures/club_face/provenance.json``).
EVENTS_S = {"address": 0.0, "top": 1.104, "impact": 1.326}
CLUBS = ("driver", "iron7")  # swing_q_<club>.npz fixtures
CLIP_FPS = 30
SPEED = 0.5
VIEWS = ("face_on", "down_the_line")
DISTANCE_M = 4.2
SIZE = (960, 720)
MAX_DRAWS = 12
DEFAULT_OUT = Path.home() / "Videos" / "Parity Audit" / "golfer_realism" / "msk_club"


def event_times_s(capture_events: Any = None) -> dict[str, float]:
    """Address, top and impact times (s) of the stills.

    The generated model's :data:`EVENTS_S` unless ``capture_events`` (a
    ``SwingEvents`` of the capture) is given.
    """
    if capture_events is None:
        return dict(EVENTS_S)
    return {
        "address": float(capture_events.address_time),
        "top": float(capture_events.top_time),
        "impact": float(capture_events.impact_time),
    }


def event_frames(n_frames: int, capture_events: Any = None) -> dict[str, int]:
    """Fixture frame index of each still (finish is the last frame).

    Event rows are clipped to the fixture; the finish is the last frame.
    """
    times = event_times_s(capture_events)
    frames = {
        k: min(n_frames - 1, int(round(t / SWING_DT_S))) for k, t in times.items()
    }
    frames["finish"] = n_frames - 1
    return frames


def check_options(club: str, mode: str, has_capture: bool) -> None:
    """Raise ``ValueError`` for an unknown club or mode, or a mode needing a capture."""
    if club not in CLUBS:
        raise ValueError(f"club must be one of {CLUBS}, got {club!r}")
    if mode not in tt.MODES:
        raise ValueError(f"mode must be one of {tt.MODES}, got {mode!r}")
    if mode != "baseline" and not has_capture:
        raise ValueError(f"--mode {mode} needs --capture")


class CaptureInputs(NamedTuple):
    """A loaded private capture and its turn inputs (never written out)."""

    capture: Any
    turn: Any

    @property
    def events(self) -> Any:
        return self.turn.events


def load_capture_inputs(capture_path: Path | None) -> CaptureInputs | None:
    """Load ``capture_path`` and its turn inputs; ``None`` without a capture."""
    if capture_path is None:
        return None
    from src.shared.python.motion_matching.tour_capture_contract import (
        load_tour_capture,
    )

    capture = load_tour_capture(capture_path)
    return CaptureInputs(capture, capture_turn_inputs(capture))


def club_model(model_path: Path, club: str, out_dir: Path) -> Path:
    """Copy of ``model_path`` holding ``club`` at its grip calibration.

    The committed models carry the driver, so the viewer draws this copy (same
    file name, written to ``out_dir``) to show the club that was tracked.
    Raises ``ValueError`` for an unknown club and ``KeyError`` when the model
    has no grip calibration for it.
    """
    if club not in CLUBS:
        raise ValueError(f"club must be one of {CLUBS}, got {club!r}")
    from defusedxml import ElementTree as SafeET

    tree = SafeET.parse(str(model_path))
    model = tree.getroot().find("Model")
    if model is None:
        raise ValueError(f"{model_path} has no <Model>")
    calibration = mc.load_calibration(model_path.stem, club)
    mc.attach_club(model, mc.load_msk_club(club), calibration)
    out = Path(out_dir) / model_path.name
    tree.write(out, encoding="utf-8", xml_declaration=True)
    return out


def clip_frames(n_frames: int) -> list[int]:
    """Fixture frames sampled so ``CLIP_FPS`` playback runs at ``SPEED``."""
    step_s = SPEED / CLIP_FPS
    count = int((n_frames - 1) * SWING_DT_S / step_s) + 1
    return [int(round(i * step_s / SWING_DT_S)) for i in range(count)]


class _Viewer:
    """Simbody visualizer of the golf humanoid on the virtual display."""

    def __init__(
        self, osim: Any, model_path: Path, geometry: Path | None, club: str
    ) -> None:
        self.osim = osim
        self.club = mc.load_msk_club(club)
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
        return cal.face_angle_deg(self.model, self.state, self.club)


def _write_png(path: Path, frame: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(frame).save(path)


def render(
    model_path: Path,
    out: Path,
    geometry: Path | None,
    *,
    club: str = "driver",
    mode: str = "baseline",
    capture_path: Path | None = None,
) -> dict[str, Any]:
    """Track the swing, write stills, clip and a JSON summary to ``out``.

    With ``capture_path`` (a private capture, never recorded in the outputs)
    the stills use the capture's events and the tracking follows ``mode``.
    """
    require_virtual_display()
    import imageio.v2 as imageio
    import opensim as osim

    check_options(club, mode, capture_path is not None)
    fixture = mc.REPO_ROOT / "tests" / "fixtures" / "club_face" / f"swing_q_{club}.npz"
    rows = np.load(fixture)["q"]
    inputs = load_capture_inputs(capture_path)
    events = event_frames(len(rows), None if inputs is None else inputs.events)
    clip = clip_frames(len(rows))
    wanted = sorted(set(clip) | set(events.values()))
    targets = feet = None
    if inputs is not None:
        targets, feet = tt.tracking_inputs(
            inputs.capture,
            np.array(wanted) * SWING_DT_S,
            mode,
            turn_inputs=inputs.turn,
        )
    frames = mt.track_swing(
        model_path, rows[wanted], club=club, turn_targets=targets, feet=feet
    )
    tracked = dict(zip(wanted, frames, strict=True))
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:  # the model is read on load
        viewer = _Viewer(osim, club_model(model_path, club, Path(tmp)), geometry, club)
    floor = cal.GeneratedSwing(club).floor_native_z
    viewer.pose(tracked[events["address"]].q)
    pelvis = viewer.model.getBodySet().get("pelvis")
    lookat = cal.transform_in_ground(pelvis, viewer.state)[:3, 3] * [1.0, 0.0, 1.0]
    lookat[1] = 0.95
    summary: dict[str, Any] = {
        "model": model_path.stem,
        "club": club,
        "mode": mode,
        "capture_used": inputs is not None,
        "events": {},
    }
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
    if inputs is not None:
        summary["msk_turn"] = _msk_turn(tracked, inputs.events, mode)
    (out / "msk_club_render_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def _msk_turn(
    tracked: dict[int, Any], capture_events: Any, mode: str
) -> dict[str, Any]:
    """Model turn lines (shared ``swing_comparison.turn``) of the tracked rows."""
    ks = sorted(tracked)
    times = np.array(ks) * SWING_DT_S
    first = tracked[ks[0]].turn_points
    points = {key: np.array([tracked[k].turn_points[key] for k in ks]) for key in first}
    lines = model_turn_lines(points, times, capture_events)
    return turn_source_block(lines, capture_events, f"opensim_msk_retarget_{mode}")


def build_parser() -> argparse.ArgumentParser:
    """Command-line parser of this script."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model", type=Path, default=mc.MODELS_DIR / "golf_humanoid.osim"
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--geometry", type=Path, default=None, help="OpenSim Geometry")
    parser.add_argument(
        "--capture", type=Path, default=None, help="private capture C3D (turn inputs)"
    )
    parser.add_argument("--club", default="driver", choices=CLUBS)
    parser.add_argument("--mode", default=None, choices=tt.MODES)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and complete the options (``mode`` and ``out`` defaults).

    A non-baseline ``--mode`` needs ``--capture`` (usage error otherwise).
    Without a capture the run is the baseline driver render of before.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mode is None:
        args.mode = "turn_feet" if args.capture is not None else "baseline"
    try:
        check_options(args.club, args.mode, args.capture is not None)
    except ValueError as exc:
        parser.error(str(exc))
    if args.out is None:
        with_capture = args.capture is not None
        args.out = (
            DEFAULT_OUT / f"{args.club}_{args.mode}" if with_capture else DEFAULT_OUT
        )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    summary = render(
        args.model,
        args.out,
        args.geometry,
        club=args.club,
        mode=args.mode,
        capture_path=args.capture,
    )
    logger.info("%s", json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
