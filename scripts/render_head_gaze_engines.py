"""Head-gaze clips in the native viewers of Drake, Pinocchio, OpenSim, MyoSuite (OSV-3b).

For one pipeline/sweep run directory this builds a same-input bundle from the
run's inverse-kinematics reference ``q_ref`` (so every engine replays the same
``q`` and the visible head follows the fitted neck), then exports the native
viewer clips with two extra glyphs from the eye point: the head-forward axis
(red) and the line of sight to the ball at address (green). Run it once per
gaze weight, then pair the clips with ``--pair``.

    MUJOCO_GL=egl python3 -m scripts.render_head_gaze_engines RUN OUT \\
        --label driver_gaze0 --engines drake,pinocchio
    python3 -m scripts.render_head_gaze_engines --pair OFF_DIR ON_DIR OUT \\
        --capture driver
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

GLYPH_LENGTH_M = 0.55
SHAFT_RADIUS_M = 0.006
CONE_LENGTH_M = 0.07
HEAD_FORWARD_RGBA = (0.95, 0.1, 0.1, 1.0)
SIGHT_RGBA = (0.2, 0.9, 0.3, 1.0)
# One panel of the gaze off | gaze on pair, so the paired clip is 1920x1080.
PANEL_WIDTH = 960
PANEL_HEIGHT = 1080
LOG = logging.getLogger("head_gaze_engines")


def head_glyph_arrows(
    eye: Sequence[float], forward: Sequence[float], ball: Sequence[float]
) -> tuple[Any, ...]:
    """Head-forward and line-of-sight ``ArrowGlyph`` pair from the eye point."""
    from src.shared.python.force_overlay.contracts import WrenchKind
    from src.shared.python.force_overlay.glyphs import ArrowGlyph

    e = np.asarray(eye, dtype=float)
    f = np.asarray(forward, dtype=float)
    b = np.asarray(ball, dtype=float)
    if not (np.isfinite(e).all() and np.isfinite(f).all() and np.isfinite(b).all()):
        raise ValueError("eye, forward and ball must be finite")
    if np.linalg.norm(f) < 1e-9 or np.linalg.norm(b - e) < 1e-9:
        raise ValueError("forward axis and sight line must be nonzero")
    out = []
    for label, vec, rgba, radius in (
        (
            "head_forward",
            f / np.linalg.norm(f) * GLYPH_LENGTH_M,
            HEAD_FORWARD_RGBA,
            0.02,
        ),
        ("line_of_sight", b - e, SIGHT_RGBA, 0.012),
    ):
        tip = e + vec
        base = tip - vec / np.linalg.norm(vec) * CONE_LENGTH_M
        out.append(
            ArrowGlyph(
                label=label,
                kind=WrenchKind.EXTERNAL,
                tail_m=tuple(float(v) for v in e),
                tip_m=tuple(float(v) for v in tip),
                head_base_m=tuple(float(v) for v in base),
                shaft_radius_m=SHAFT_RADIUS_M,
                head_radius_m=radius,
                rgba=rgba,
                magnitude=float(np.linalg.norm(vec)),
                units="m",
                clamped=False,
            )
        )
    return tuple(out)


BUNDLE_INPUTS = ("full_body_spec_hipcal_scaled.json", "ik_trajectory.npz")


def bundle_is_current(run: Path, out_npz: Path) -> bool:
    """True when ``out_npz`` exists and is newer than every bundle input in ``run``."""
    if not out_npz.is_file():
        return False
    built = out_npz.stat().st_mtime
    return all((run / name).stat().st_mtime <= built for name in BUNDLE_INPUTS)


def build_bundle(run: Path, out_npz: Path) -> Path:
    """Same-input bundle that replays the run's ``q_ref`` (closed loop in MuJoCo).

    The closed-loop replay takes minutes, so a bundle newer than its inputs
    is reused (one bundle serves every engine pass of a run).
    """
    from src.shared.python.motion_matching.same_input import (
        generate_reference_bundle,
    )

    if bundle_is_current(run, out_npz):
        LOG.info("reusing bundle %s", out_npz)
        return out_npz
    spec_bytes = (run / "full_body_spec_hipcal_scaled.json").read_bytes()
    with np.load(run / "ik_trajectory.npz") as ik:
        times, q_ref = ik["time_s"], ik["q_ref"]
    bundle = generate_reference_bundle(
        spec_bytes,
        times - times[0],
        q_ref,
        provenance={"run_dir": run.name, "source": "ik_trajectory q_ref"},
    )
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    bundle.save(out_npz)
    return out_npz


def overlay_factory_for(run: Path):  # noqa: ANN202
    """``(swing, engine) -> (OverlayFeed, lookat)`` drawing the two head glyphs."""
    from src.shared.python.force_overlay.glyphs import GlyphSet, LegendSpec
    from src.shared.python.motion_matching import gaze
    from src.shared.python.motion_matching.pipeline import gaze_residual as gr
    from src.shared.python.motion_matching.pipeline.plant import get_plant
    from src.tools.native_viewer_export.core import OverlayFeed, default_glyph_style

    spec = json.loads((run / "full_body_spec_hipcal_scaled.json").read_text())
    sweep = json.loads((run / "sweep_row.json").read_text())["head_gaze"]
    plan = sweep["plan"]
    ball = np.asarray(plan["ball_at_address_m"], dtype=float)
    axis = np.asarray(plan["gaze_axis_head"], dtype=float)
    attachments = {
        k: (v["body"], tuple(v["offset_m"]))
        for k, v in spec["marker_attachments"].items()
        if v["offset_m"] is not None
    }
    kin = get_plant("mujoco", spec).create_ik(dict(list(attachments.items())[:5]))
    if tuple(kin.coordinate_order) != tuple(spec["coordinate_order"]):
        raise ValueError("kinematics coordinate order differs from the spec")

    def factory(swing, _engine):  # noqa: ANN001, ANN202
        rot, trans = gr.frame_poses(kin, swing.q, gr.HEAD_FRAME)
        eyes = gaze.eye_point(rot, trans)
        forward = gaze.gaze_direction(rot, axis)

        def build(index: int, _style: Any) -> GlyphSet:
            arrows = head_glyph_arrows(eyes[index], forward[index], ball)
            return GlyphSet(
                time_s=float(index), arrows=arrows, torque_arcs=(), legend=LegendSpec()
            )

        lookat = (float(eyes[0][0]), float(eyes[0][1]), 0.9)
        return OverlayFeed(lambda i: i, default_glyph_style(70.0), build), lookat

    return factory


def export_run(run: Path, out: Path, label: str, engines: Sequence[str]) -> None:
    from src.tools.native_viewer_export.core import ExportSettings
    from src.tools.native_viewer_export.runner import ExportJob, run_export

    bundle = build_bundle(run, out / "_bundles" / f"{label}.npz")
    plan = json.loads((run / "sweep_row.json").read_text())["head_gaze"]["plan"]
    with np.load(run / "ik_trajectory.npz") as ik:
        impact_s = float(plan["impact_time_s"] - ik["time_s"][0])
    settings = replace(
        ExportSettings(),
        width=PANEL_WIDTH,
        height=PANEL_HEIGHT,
        fps=60,
        speeds=(1.0, 0.5),
        impact_time_s=impact_s,
        impact_window_s=0.4,
        impact_speed=0.25,
        views=("face_on", "down_the_line"),
        multiview=False,
    )
    job = ExportJob(bundle, out / label, label, label, tuple(engines))
    for result in run_export(job, settings, overlay_factory=overlay_factory_for(run)):
        LOG.info("%s: %s", result.engine, result.skipped_reason or result.paths)


def pair_clips(off_dir: Path, on_dir: Path, out: Path, capture: str) -> list[Path]:
    """Side-by-side (gaze off | gaze on) clip per engine and view, same frame count."""
    import cv2
    import imageio.v2 as imageio

    written = []
    for clip_off in sorted(off_dir.glob("*.mp4")):
        clip_on = on_dir / clip_off.name.replace("gaze0", "gazeon")
        if not clip_on.exists():
            LOG.warning("no matching gaze-on clip for %s", clip_off.name)
            continue
        dest = out / clip_off.name.replace("gaze0", "gaze_off_vs_on")
        out.mkdir(parents=True, exist_ok=True)
        with (
            imageio.get_reader(str(clip_off)) as ra,
            imageio.get_reader(str(clip_on)) as rb,
            imageio.get_writer(
                str(dest), fps=ra.get_meta_data()["fps"], codec="libx264", quality=7
            ) as writer,
        ):
            for fa, fb in zip(ra, rb, strict=False):
                for frame, text in ((fa, f"{capture}: gaze off"), (fb, "gaze on")):
                    cv2.putText(
                        frame, text, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (255, 255, 255), 2, cv2.LINE_AA,
                    )  # fmt: skip
                writer.append_data(np.hstack([fa, fb]))
        written.append(dest)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--label", default="run")
    parser.add_argument("--engines", default="drake,pinocchio,opensim,myosuite")
    parser.add_argument("--pair", action="store_true")
    parser.add_argument("--capture", default="driver")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.pair:
        off, on, out = args.paths
        pair_clips(off, on, out, args.capture)
    else:
        run, out = args.paths
        export_run(run, out, args.label, tuple(args.engines.split(",")))


if __name__ == "__main__":
    main()
