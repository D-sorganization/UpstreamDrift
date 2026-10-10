"""Annotated address stills (overhead, face-on) in each engine's native viewer.

OSV-4 (#11730). Reuses ``src.tools.native_viewer_export`` backends (Drake and
Pinocchio MeshCat, OpenSim simbody, MyoSuite arena, MuJoCo Renderer) and the
``golf_view_presets`` cameras; the fitted address ``q`` of an
``address_report.json`` run directory is shown as a one-state swing. Each still
is annotated with every foot's model toe-out against the capture target, read
from the ``engines.json`` written by ``scripts.address_foot_progression_engines``:

    python3 -m scripts.render_address_engine_stills --run-dir RUN \
        --engines-json RUN/engines.json --engines drake pinocchio --out DIR

An engine whose native viewer is unavailable here is reported, never faked.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)
VIEWS = ("overhead", "face_on")
SIZE = (960, 720)
SCALED_SPEC = "full_body_spec_hipcal_scaled.json"
REPORT = "address_report.json"


def annotation_lines(engine: str, label: str, doc: Mapping[str, Any]) -> list[str]:
    """Text lines: model vs capture toe-out per foot for ``engine``.

    Raises ``ValueError`` when the engine entry is missing or unavailable.
    """
    entry = doc.get("engines", {}).get(engine)
    if not entry or not entry.get("available"):
        raise ValueError(f"{engine} has no available foot progression entry")
    lines = [f"{engine} / {label}: address toe-out (deg), model vs capture target"]
    for side, role in (("left", "lead"), ("right", "trail")):
        model = entry["model_deg"][side]
        target = doc["targets_deg"][side]
        err = entry["error_deg"][side]
        lines.append(
            f"{role} ({side}): model {model:+.1f}  capture {target:+.1f}  "
            f"error {err:+.1f}"
        )
    return lines


def annotate(image: np.ndarray, lines: Sequence[str]) -> np.ndarray:
    """Black-backed white text block at the top left of an RGB frame."""
    from PIL import Image, ImageDraw

    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    y = 8
    for line in lines:
        draw.rectangle((4, y - 2, 4 + 7 * len(line), y + 13), fill=(0, 0, 0))
        draw.text((8, y), line, fill=(255, 255, 255))
        y += 18
    return np.asarray(pil)


def draw_foot_axes(
    image: np.ndarray,
    view: str,
    look: Sequence[float],
    distance_m: float,
    points: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Project each foot's calcn -> toes axis (red) and the straight-ahead
    reference (white) through the shared view camera onto ``image``."""
    import cv2

    from src.shared.python.golf_view_presets import VIEWER_FOV_Y_RAD
    from src.shared.python.motion_matching.foot_progression import model_long_axis
    from src.shared.python.motion_matching.pipeline.address_feet import (
        NATIVE_TARGET_AXIS,
        NATIVE_UP_AXIS,
    )
    from src.tools.native_viewer_export.overlay2d import pinhole_for_view

    height, width = image.shape[:2]
    cam = pinhole_for_view(view, look, distance_m, VIEWER_FOV_Y_RAD, (width, height))
    rot = np.asarray(cam.rotation_world_from_camera)
    origin = np.asarray(cam.translation_world_from_camera_m)
    k = np.asarray(cam.matrix)
    forward = np.cross(NATIVE_TARGET_AXIS, NATIVE_UP_AXIS)

    def pix(p: np.ndarray) -> tuple[int, int]:
        c = rot.T @ (p - origin)
        uv = k @ c
        return int(round(uv[0] / uv[2])), int(round(uv[1] / uv[2]))

    out = np.ascontiguousarray(image.copy())
    for sfx in ("l", "r"):
        calcn, toes = points[f"calcn_{sfx}"], points[f"toes_{sfx}"]
        axis = model_long_axis(calcn, toes, NATIVE_UP_AXIS)
        base = calcn + np.array([0.0, 0.0, 0.02])
        cv2.line(out, pix(base), pix(base + 0.4 * forward), (255, 255, 255), 2)
        cv2.line(out, pix(base), pix(base + 0.4 * axis), (230, 20, 20), 3)
        tip = pix(base + 0.4 * axis)
        role = "lead" if sfx == "l" else "trail"
        cv2.putText(
            out,
            role,
            (tip[0] + 6, tip[1] + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return out


def _swing(run_dir: Path) -> Any:
    from src.shared.python.motion_matching.same_input import InputBundle
    from src.tools.native_viewer_export.core import SwingInput

    spec_bytes = (run_dir / SCALED_SPEC).read_bytes()
    spec = json.loads(spec_bytes)
    report = json.loads((run_dir / REPORT).read_text(encoding="utf-8"))
    angles = report["address_coordinates_deg"]
    shifts = report["address_translations_m"]
    names = tuple(spec["coordinate_order"])
    q = np.array(
        [shifts[n] if n in shifts else np.radians(angles[n]) for n in names],
        dtype=float,
    )
    stack = np.vstack([q, q, q])
    bundle = InputBundle(
        spec_bytes=spec_bytes,
        coordinate_order=names,
        dt_s=0.001,
        q0=q,
        v0=np.zeros_like(q),
        efforts=np.zeros((2, len(names))),
        reference_q=stack,
        reference_v=np.zeros_like(stack),
        reference_engine="mujoco",
    )
    return SwingInput(bundle, stack, "address", "Driver", "mujoco")


def render_engine(
    engine: str,
    run_dir: Path,
    doc: Mapping[str, Any],
    out_dir: Path,
    label: str,
) -> list[Path]:
    """Write ``<label>_<engine>_<view>.png`` for the overhead and face-on views."""
    import imageio.v2 as imageio

    from src.tools.native_viewer_export.backends.registry import make_backend
    from src.tools.native_viewer_export.core import ExportSettings

    lines = annotation_lines(engine, label, doc)
    backend = make_backend(engine)
    reason = backend.unavailable_reason()
    if reason is not None:
        raise RuntimeError(f"{engine} native viewer unavailable: {reason}")
    points = {
        k: np.asarray(v) for k, v in doc["engines"][engine]["leg_points_m"].items()
    }
    heels = [points["calcn_l"], points["calcn_r"]]
    mid = 0.5 * (heels[0] + heels[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for view in VIEWS:
        look = (
            (float(mid[0]), float(mid[1]), 0.05)
            if view == "overhead"
            else (float(mid[0]), float(mid[1]), 0.85)
        )
        settings = ExportSettings(
            views=(view,),
            width=SIZE[0],
            height=SIZE[1],
            lookat_m=look,
            distance_m=2.6 if view == "overhead" else 3.6,
            multiview=False,
            overlays=False,
            hud=False,
        )
        frame = next(iter(backend.render(_swing(run_dir), settings, [0], None)))[view]
        dist = settings.distance_m
        frame = draw_foot_axes(np.asarray(frame), view, look, dist, points)
        path = out_dir / f"{label}_{engine}_{view}.png"
        imageio.imwrite(path, annotate(frame, lines))
        paths.append(path)
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--engines-json", type=Path, required=True)
    parser.add_argument("--engines", nargs="+", required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    doc = json.loads(args.engines_json.read_text(encoding="utf-8"))
    label = args.label or doc["label"]
    failed = 0
    for engine in args.engines:
        try:
            for path in render_engine(engine, args.run_dir, doc, args.out, label):
                LOGGER.info("%s", path)
        except (RuntimeError, ValueError) as exc:
            failed += 1
            LOGGER.error("%s: %s", engine, exc)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
