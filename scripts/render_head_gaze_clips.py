"""Side-by-side head-gaze clips (OSV-3, #11729), MuJoCo, headless, 1080p60.

Left: gaze weight 0 (marker-faithful). Right: gaze on. Each panel draws a red
head-forward glyph from the eye point along the gaze axis (green: line of sight
to the ball), so head orientation is visible without a face mesh, and the ball
at address (white). Inputs are two pipeline run directories
(``ik_trajectory.npz`` + ``full_body_spec_hipcal_scaled.json``).

Output is the fleet video standard: two 960x1080 panels side by side (1920x1080)
at 60 fps, one clip per ``--speeds`` entry plus an impact-centred slow clip,
sampled by time with the shared GCV-14 ``FrameSchedule``::

    MUJOCO_GL=egl python3 -m scripts.render_head_gaze_clips RUN_W0 RUN_ON OUT_STEM

writes ``OUT_STEM_1x.mp4``, ``OUT_STEM_0p5x.mp4`` and
``OUT_STEM_impact_0p25x.mp4``. ``--trajectory replay`` renders each run's
forward-dynamics replay (``dynamics_record.npz``) instead of ``q_ref``, for
example the IK neck against the gaze-schedule neck (``--fd-neck gaze``, OSV-3c)
with ``--labels "IK neck" "gaze neck"``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from src.shared.python.contracts import require
from src.shared.python.video_timing.frame_schedule import (
    DEFAULT_FPS,
    FrameSchedule,
    speed_suffix,
)

PANEL_HEIGHT, PANEL_WIDTH = 1080, 960
GLYPH_LENGTH_M = 0.6
# mjtJoint values: the IK coordinates are scalar joints, so no quaternion slerp.
_HINGE, _SLIDE = 3, 2
#: ``ik``: the reference ``q_ref``; ``replay``: the forward-dynamics replay.
TRAJECTORIES = ("ik", "replay")


def replay_trajectory(run: Path) -> tuple[np.ndarray, np.ndarray]:
    """Forward-dynamics replay ``(q, time_s)`` of a run, on the capture times.

    ``dynamics_record.npz`` samples the simulation on its own clock; the state
    is interpolated onto ``track_time_s`` (the capture times of ``q_track``).
    """
    rec = np.load(run / "dynamics_record.npz")
    times = np.asarray(rec["track_time_s"], dtype=float)
    sim_t, sim_q = np.asarray(rec["time_s"]), np.asarray(rec["q"])
    require(sim_q.ndim == 2 and len(sim_t) == len(sim_q), "malformed dynamics record")
    q = np.column_stack([np.interp(times, sim_t, col) for col in sim_q.T])
    return q, times


def _load(run: Path, trajectory: str = "ik"):  # noqa: ANN202
    require(trajectory in TRAJECTORIES, f"trajectory must be one of {TRAJECTORIES}")
    spec_bytes = (run / "full_body_spec_hipcal_scaled.json").read_bytes()
    q = np.load(run / "ik_trajectory.npz")
    # A full pipeline run writes receipt.json; a sweep run (scripts.sweep_gaze_weight)
    # stops after the inverse kinematics and writes sweep_row.json instead.
    receipt_path = run / "receipt.json"
    if receipt_path.exists():
        block = json.loads(receipt_path.read_text(encoding="utf-8"))["head_gaze"]
    else:
        block = json.loads((run / "sweep_row.json").read_text(encoding="utf-8"))[
            "head_gaze"
        ]
    if trajectory == "replay":
        return spec_bytes, *replay_trajectory(run), block
    return spec_bytes, q["q_ref"], q["time_s"], block


def _connector(scene, p0, p1, radius, rgba) -> None:  # noqa: ANN001
    import mujoco

    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        radius,
        np.asarray(p0, dtype=float),
        np.asarray(p1, dtype=float),
    )
    scene.ngeom += 1


def clip_schedules(
    times_s: np.ndarray,
    impact_time_s: float,
    speeds: tuple[float, ...],
    impact_speed: float,
    impact_window_s: float,
    fps: float = DEFAULT_FPS,
) -> list[tuple[str, FrameSchedule]]:
    """``(file suffix, schedule)`` per full-swing speed plus the impact clip.

    The impact clip shows ``impact_time_s +- impact_window_s / 2`` (clipped to
    the data) at ``impact_speed``.

    Raises:
        ValueError: on empty ``speeds``, a non-positive speed or window, or an
            impact time outside the sampled range.
    """
    t = np.asarray(times_s, dtype=float)
    require(len(speeds) > 0, "speeds must not be empty")
    require(all(s > 0.0 for s in speeds), "speeds must be positive")
    require(impact_speed > 0.0, "impact_speed must be positive")
    require(impact_window_s > 0.0, "impact_window_s must be positive")
    require(
        bool(t[0] <= impact_time_s <= t[-1]), "impact time outside the sampled range"
    )
    plan = [(speed_suffix(s), FrameSchedule(t, fps, s)) for s in speeds]
    half = 0.5 * impact_window_s
    window = (impact_time_s - half, impact_time_s + half)
    impact = FrameSchedule(t, fps, impact_speed, window)
    plan.append(("_impact" + speed_suffix(impact_speed), impact))
    return plan


def _panel_renderer(  # noqa: ANN202
    run: Path, label: str, azimuth: float, trajectory: str = "ik"
):
    """Load ``run`` once; return ``(render, times, impact_t)``.

    ``render(schedule)`` yields one panel image per scheduled frame, rendered
    lazily so a clip never holds all of its frames in memory.
    """
    import cv2
    import mujoco

    from src.engines.physics_engines.mujoco.python import full_body_mjcf as exporter
    from src.engines.physics_engines.mujoco.python.visual_layer import (
        add_scene_marker,
    )
    from src.shared.python.motion_matching import gaze
    from src.shared.python.motion_matching.pipeline import gaze_residual as gr
    from src.shared.python.motion_matching.pipeline.plant import get_plant

    spec_bytes, q_src, times, block = _load(run, trajectory)
    spec = json.loads(spec_bytes)
    plant = get_plant("mujoco", spec)
    att = {
        k: (v["body"], tuple(v["offset_m"]))
        for k, v in spec["marker_attachments"].items()
        if v["offset_m"] is not None
    }
    kin = plant.create_ik(dict(list(att.items())[:5]), ik_backend="lm")
    ball = np.asarray(block["plan"]["ball_at_address_m"])
    impact_t = float(block["plan"]["impact_time_s"])  # absolute, like ``times``
    hold_s = 3.0 * float(np.median(np.diff(times)))  # "ball" phase ends just after

    xml, _ = exporter.export_full_body_mjcf(spec_bytes, visual=True)
    model = mujoco.MjModel.from_xml_string(xml)
    # The offscreen framebuffer defaults to 640x480; it must hold a whole panel.
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, PANEL_WIDTH)
    model.vis.global_.offheight = max(model.vis.global_.offheight, PANEL_HEIGHT)
    data = mujoco.MjData(model)
    names = tuple(kin.coordinate_order)
    addresses = [model.joint(n).qposadr[0] for n in names]
    kinds = {int(model.jnt_type[model.joint(n).id]) for n in names}
    require(kinds <= {_HINGE, _SLIDE}, "quaternion joints need slerp interpolation")
    renderer = mujoco.Renderer(model, PANEL_HEIGHT, PANEL_WIDTH)
    # Frame every clip on the address pose so the impact clip matches the others.
    address_r, address_t = gr.frame_poses(kin, q_src[:1], gr.HEAD_FRAME)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = 0.5 * (gaze.eye_point(address_r, address_t)[0] + ball) - np.array(
        [0.0, 0.0, 0.1]
    )
    cam.distance, cam.azimuth, cam.elevation = 3.4, azimuth, -10.0

    def render(schedule: FrameSchedule) -> Iterator[np.ndarray]:
        q = schedule.interpolate(q_src)
        head_r, head_t = gr.frame_poses(kin, q, gr.HEAD_FRAME)
        eyes = gaze.eye_point(head_r, head_t)
        forward = gaze.gaze_direction(head_r, block["plan"]["gaze_axis_head"])
        for k, t in enumerate(schedule.sample_times_s):
            data.qpos[addresses] = q[k]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            add_scene_marker(renderer.scene, ball, 0.021335, (1, 1, 1, 1))
            tip = eyes[k] + GLYPH_LENGTH_M * forward[k]
            _connector(renderer.scene, eyes[k], tip, 0.008, (0.95, 0.1, 0.1, 1))
            _connector(renderer.scene, eyes[k], ball, 0.003, (0.2, 0.9, 0.3, 1))
            add_scene_marker(renderer.scene, eyes[k], 0.014, (1.0, 0.8, 0.1, 1))
            img = np.ascontiguousarray(renderer.render())
            phase = "ball" if t <= impact_t + hold_s else "release"
            text = f"{label}  t={t - impact_t:+.2f}s  ({phase})  {schedule.speed:g}x"
            cv2.putText(
                img,
                text,
                (20, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            yield img

    return render, times, impact_t


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_w0", type=Path)
    parser.add_argument("run_on", type=Path)
    parser.add_argument("out_stem", type=Path, help="clips are <stem>_<speed>.mp4")
    parser.add_argument("--azimuth", type=float, default=20.0)
    parser.add_argument("--speeds", default="1,0.5", help="comma-separated speeds")
    parser.add_argument("--impact-speed", type=float, default=0.25)
    parser.add_argument("--impact-window", type=float, default=0.4, help="total, s")
    parser.add_argument(
        "--trajectory",
        choices=TRAJECTORIES,
        default="ik",
        help="ik: reference q_ref; replay: forward-dynamics replay (OSV-3c)",
    )
    parser.add_argument(
        "--labels", nargs=2, default=("gaze weight 0", "gaze on"), metavar=("L", "R")
    )
    args = parser.parse_args()

    import imageio

    left, times, impact_t = _panel_renderer(
        args.run_w0, args.labels[0], args.azimuth, args.trajectory
    )
    right, _, _ = _panel_renderer(
        args.run_on, args.labels[1], args.azimuth, args.trajectory
    )
    speeds = tuple(float(s) for s in args.speeds.split(","))
    plan = clip_schedules(
        times, impact_t, speeds, args.impact_speed, args.impact_window
    )
    args.out_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix, schedule in plan:
        out = args.out_stem.with_name(args.out_stem.name + suffix + ".mp4")
        # macro_block_size=1: keep exactly 1920x1080 (default pads height to 1088).
        with imageio.get_writer(
            out, fps=schedule.fps, codec="libx264", quality=7, macro_block_size=1
        ) as writer:
            for a, b in zip(left(schedule), right(schedule), strict=True):
                writer.append_data(np.hstack([a, b]))


if __name__ == "__main__":
    main()
