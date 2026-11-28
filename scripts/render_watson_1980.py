"""Render before/after native Watson appearance at desktop and phone sizes."""

import argparse
import copy
import json
import tempfile
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np

from scripts.export_player_models import _bounds, _load_scene
from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
    export_full_body_mjcf,
)
from src.shared.python.model_appearance.player_models import (
    build_player_spec,
    player_profile,
)
from src.shared.python.model_appearance.schema import (
    document_from_dict,
    document_to_dict,
)


def main() -> None:
    """Render the same sized plant before and after the 1980 appearance changes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    output = parser.parse_args().out
    output.mkdir(parents=True, exist_ok=True)
    spec = build_player_spec("tom_watson")
    raw = json.dumps(spec).encode()
    current = player_profile("tom_watson")["appearance"]
    before = copy.deepcopy(current)
    before.pop("body_build")
    before["materials"]["polo_white"]["base_color"] = [0.18, 0.29, 0.42, 1.0]
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        for label, data in (("before", before), ("after", current)):
            doc = document_from_dict(data)
            xml, _ = export_full_body_mjcf(raw, appearance=doc)
            path = folder / "model.xml"
            path.write_text(xml)
            (folder / "model.appearance.json").write_text(
                json.dumps(document_to_dict(doc))
            )
            model, state, groups, _, _, _, _, azimuth = _load_scene(path, spec, None)
            person, props, helpers, _ = groups
            model.geom_rgba[np.asarray([*props, *helpers], dtype=int), 3] = 0
            low, high = _bounds(model, state, mujoco, np.asarray(person))
            model.vis.global_.offwidth, model.vis.global_.offheight = 1280, 960
            for width, height in ((1280, 960), (390, 844)):
                renderer = mujoco.Renderer(model, height, width)
                try:
                    camera = mujoco.MjvCamera()
                    camera.lookat[:] = (low + high) / 2
                    camera.distance = max((high - low)[2] * 1.55, 3.0)
                    camera.azimuth, camera.elevation = azimuth, -5
                    renderer.update_scene(state, camera=camera)
                    iio.imwrite(
                        output / f"{label}_{width}.png", renderer.render().copy()
                    )
                finally:
                    renderer.close()


if __name__ == "__main__":
    main()
