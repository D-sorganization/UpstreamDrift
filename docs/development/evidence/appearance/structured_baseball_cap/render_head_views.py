"""Render head-focused views from an existing appearance-only model export."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v3 as iio
import mujoco
import numpy as np


def main() -> None:
    """Render front, three-quarter and profile head crops for desktop and phone."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo, model_dir = args.repo.resolve(), args.model_dir.resolve()
    sys.path.insert(0, str(repo))

    from scripts.export_player_models import _bounds, _load_scene

    xml_path = model_dir / "model.xml"
    spec = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
    model, data, groups, *_rest, front = _load_scene(xml_path, spec, None)
    _person, props, helpers, head = groups
    model.geom_rgba[np.asarray([*props, *helpers], dtype=int), 3] = 0
    low, high = _bounds(model, data, mujoco, np.asarray(head, dtype=int))
    center, extent = (low + high) / 2, high - low
    distance = max(float(extent.max()) * 2.5, 0.35)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for width, height in ((1280, 960), (390, 844)):
        model.vis.global_.offwidth, model.vis.global_.offheight = width, height
        with mujoco.Renderer(model, height, width) as renderer:
            for label, offset in (("front", 0), ("threequarter", 35), ("profile", 90)):
                camera = mujoco.MjvCamera()
                camera.lookat[:] = center
                camera.distance = distance
                camera.azimuth = (front + offset) % 360
                camera.elevation = -5
                renderer.update_scene(data, camera=camera)
                target = args.output_dir / f"{label}_head_{width}x{height}.png"
                iio.imwrite(target, renderer.render().copy())


if __name__ == "__main__":
    main()
