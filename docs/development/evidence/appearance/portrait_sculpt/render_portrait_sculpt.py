"""Recreate a restrained before/after portrait render at desktop and phone sizes."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import zipfile
from pathlib import Path

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v3 as iio
import mujoco
import numpy as np


def main() -> None:
    """Render one archive-backed portrait with native viewport dimensions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo, out = args.repo.resolve(), args.out.resolve()
    sys.path.insert(0, str(repo))

    from src.shared.python.model_appearance import document_from_dict
    from src.shared.python.model_appearance.schema import save_appearance
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )
    from scripts.export_player_models import _bounds, _load_scene

    archive_path = repo / "assets/player_models/legends/legends-models.zip"
    with zipfile.ZipFile(archive_path) as archive:
        prefix = "sized/seve_ballesteros/"
        spec_bytes = archive.read(prefix + "model.json")
        source_appearance = json.loads(archive.read(prefix + "model.appearance.json"))
        source_xml = archive.read(prefix + "model.xml").decode("utf-8")

    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="portrait-sculpt-") as temporary:
        work = Path(temporary)
        for label in ("before", "after"):
            folder = work / label
            folder.mkdir()
            preview_folder = out / label
            preview_folder.mkdir(parents=True, exist_ok=True)
            document = json.loads(json.dumps(source_appearance))
            if label == "after":
                document["head"]["morph"].update(
                    width=-0.04,
                    jaw=0.02,
                    nose=0.16,
                    mouth=0.05,
                    eyes=-0.06,
                    cheeks=0.04,
                    chin=0.12,
                    brow=0.18,
                    nose_bridge=0.18,
                    nose_length=0.10,
                    nose_width=-0.12,
                    eye_opening=-0.14,
                    face_length=0.12,
                    ears=0.02,
                )
                document["head"]["facial_hair"] = "moustache"
            appearance = document_from_dict(document)
            save_appearance(appearance, folder / "model.appearance.json")
            (folder / "model.json").write_bytes(spec_bytes)
            xml = (
                source_xml
                if label == "before"
                else export_full_body_mjcf(spec_bytes, appearance=appearance)[0]
            )
            xml_path = folder / "model.xml"
            xml_path.write_text(xml, encoding="utf-8")
            spec = json.loads(spec_bytes)
            model, data, groups, _fit, *_rest, front = _load_scene(xml_path, spec, None)
            person, props, helpers, head = groups
            head_low, head_high = _bounds(
                model, data, mujoco, np.asarray(head, dtype=int)
            )
            center = (head_low + head_high) / 2
            extent = head_high - head_low
            model.geom_rgba[np.asarray([*props, *helpers], dtype=int), 3] = 0

            model.vis.global_.fovy = 45
            for width, height in ((1280, 960), (390, 844)):
                model.vis.global_.offwidth = width
                model.vis.global_.offheight = height
                with mujoco.Renderer(model, height, width) as renderer:
                    for view, offset in (
                        ("front", 0),
                        ("threequarter", 35),
                        ("profile", 90),
                    ):
                        azimuth = (front + offset) % 360
                        if width == 390:
                            radius = float(np.linalg.norm(extent) / 2)
                            distance = max(
                                radius
                                / (math.tan(math.radians(45 / 2)) * width / height)
                                * 1.12,
                                0.35,
                            )
                        else:
                            distance = max(float(extent.max()) * 2.5, 0.35)
                        camera = mujoco.MjvCamera()
                        camera.lookat[:] = center
                        camera.distance = distance
                        camera.azimuth = azimuth
                        camera.elevation = -5
                        renderer.update_scene(data, camera=camera)
                        iio.imwrite(
                            preview_folder / f"{view}_{width}x{height}.png",
                            renderer.render().copy(),
                        )


if __name__ == "__main__":
    main()
