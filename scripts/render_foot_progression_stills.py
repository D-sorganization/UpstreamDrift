"""Render annotated address stills (overhead and face-on) of a fitted full-body model.

OSV-4 (#11730). MuJoCo only (the shared visual export); other engines render
their own stills from the same address ``q``. Each foot's calcn -> toes axis is
drawn in red, the straight-ahead reference in white, and the angles (model
versus capture/target) are written on the image:

    python3 -m scripts.render_foot_progression_stills --spec spec.json \
        --address address.npz --probe probe.json --label after --out DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.golf_view_presets import get_view_preset, mujoco_camera_params
from src.shared.python.motion_matching.foot_progression import model_long_axis
from src.shared.python.motion_matching.pipeline.address_feet import (
    NATIVE_TARGET_AXIS,
    NATIVE_UP_AXIS,
)

VIEWS = ("overhead", "face_on")
SIZE = (720, 960)
LINE_LENGTH_M = 0.40
AXIS_RGBA = (0.95, 0.1, 0.1, 1.0)
FORWARD_RGBA = (1.0, 1.0, 1.0, 1.0)


def _add_line(scene: Any, start: np.ndarray, end: np.ndarray, rgba: tuple) -> None:
    import mujoco

    if scene.ngeom >= scene.maxgeom:
        raise ValueError("scene has no free geom slot")
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),
        np.zeros(3),
        np.zeros(9),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.006, start, end)
    scene.ngeom += 1


def _foot_points(model: Any, data: Any) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    import mujoco

    out = {}
    for side in ("r", "l"):
        pts = []
        for body in (f"calcn_{side}", f"toes_{side}"):
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
            pts.append(np.asarray(data.xpos[body_id], dtype=float).copy())
        out[side] = (pts[0], pts[1])
    return out


def _annotate(image: np.ndarray, lines: list[str]) -> np.ndarray:
    from PIL import Image, ImageDraw

    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    y = 10
    for line in lines:
        draw.rectangle((6, y - 2, 6 + 8 * len(line), y + 14), fill=(0, 0, 0))
        draw.text((10, y), line, fill=(255, 255, 255))
        y += 18
    return np.asarray(pil)


def render(
    spec_path: Path, address_path: Path, probe_path: Path, label: str, out_dir: Path
) -> list[Path]:
    """Write ``<label>_overhead.png`` and ``<label>_face_on.png`` into ``out_dir``."""
    import imageio.v2 as imageio
    import mujoco

    from src.engines.physics_engines.mujoco.python import full_body_mjcf as exporter

    spec_bytes = spec_path.read_bytes()
    saved = np.load(address_path, allow_pickle=False)
    probe = json.loads(probe_path.read_text())
    xml, _ = exporter.export_full_body_mjcf(spec_bytes, visual=True)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    for name, value in zip(saved["coords"], saved["q"], strict=True):
        data.qpos[model.joint(str(name)).qposadr[0]] = value
    mujoco.mj_forward(model, data)
    feet = _foot_points(model, data)
    forward = np.cross(NATIVE_TARGET_AXIS, NATIVE_UP_AXIS)
    lookat = 0.5 * (feet["r"][0] + feet["l"][0])
    lookat[2] = 0.9
    lines = [f"{label}: address foot progression (toe-out, deg)"]
    for side, role in (("l", "lead"), ("r", "trail")):
        key = "left" if side == "l" else "right"
        target = probe["capture_targets"][key]
        lines.append(
            f"{role} ({key}): model {probe['model'][key]:+.1f}  capture {target:+.1f}"
        )
    renderer = mujoco.Renderer(model, *SIZE)
    paths: list[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for view in VIEWS:
        cam_params = mujoco_camera_params(
            get_view_preset(view), lookat, 2.2 if view == "face_on" else 2.6
        )
        cam = mujoco.MjvCamera()
        cam.lookat[:] = cam_params.lookat
        cam.distance, cam.azimuth, cam.elevation = (
            cam_params.distance,
            cam_params.azimuth,
            cam_params.elevation,
        )
        renderer.update_scene(data, camera=cam)
        for calcn, toes in feet.values():
            axis = model_long_axis(calcn, toes, NATIVE_UP_AXIS)
            base = calcn + np.array([0.0, 0.0, 0.02])
            _add_line(renderer.scene, base, base + LINE_LENGTH_M * axis, AXIS_RGBA)
            _add_line(
                renderer.scene, base, base + LINE_LENGTH_M * forward, FORWARD_RGBA
            )
        image = _annotate(renderer.render().copy(), lines)
        path = out_dir / f"{label}_{view}.png"
        imageio.imwrite(path, image)
        paths.append(path)
    return paths


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--address", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    for path in render(args.spec, args.address, args.probe, args.label, args.out):
        print(path)  # noqa: T201 - CLI output
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
