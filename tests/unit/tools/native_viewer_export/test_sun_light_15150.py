"""Directional sun that lights the far ground of the MuJoCo export (#15150)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.tools.native_viewer_export.backends import _sun_light as sun_light
from src.tools.native_viewer_export.backends._sun_light import (
    SunLight,
    add_sun,
    load_sun_light,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

WORLD = """<mujoco><visual><headlight ambient="0.3 0.3 0.3" diffuse="0 0 0"
 specular="0 0 0"/><map zfar="400"/></visual><worldbody>
<light name="spot" pos="0 0 3" dir="0 0 -1" diffuse="0.4 0.4 0.4" cutoff="20"/>
<geom name="floor" type="plane" size="400 400 0.1" rgba="0.3 0.5 0.3 1"/>
<geom type="sphere" size="0.2" pos="0 0 0.2"/></worldbody></mujoco>"""


def _sun(**changes: object) -> SunLight:
    base = load_sun_light()
    return SunLight(**{**base.__dict__, **changes})  # type: ignore[arg-type]


def test_defaults_are_a_shadowless_downward_sun() -> None:
    sun = load_sun_light()
    assert sun.enabled and not sun.castshadow and sun.direction[2] < 0.0
    raw = json.loads(Path(sun_light.SETTINGS_PATH).read_text(encoding="utf-8"))
    assert set(raw["sources"]) == set(raw) - {"sources"}


def test_sun_is_a_directional_light_with_the_settings() -> None:
    mujoco = pytest.importorskip("mujoco")
    sun = _sun(direction=(1.0, 0.0, -1.0), intensity=0.5, castshadow=True)
    model = mujoco.MjModel.from_xml_string(add_sun(WORLD, sun))
    light = model.light(sun_light.LIGHT_NAME)
    assert int(model.light_type[light.id]) == int(
        mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    )
    assert int(light.castshadow[0]) == 1
    assert np.allclose(light.dir, np.array([1.0, 0.0, -1.0]) / np.sqrt(2.0))
    assert np.allclose(light.diffuse, np.array(sun.color) * 0.5)
    assert np.allclose(light.specular, 0.0)


def test_disabled_sun_leaves_the_model_and_invalid_settings_fail() -> None:
    assert add_sun(WORLD, _sun(enabled=False)) == WORLD
    for changes in ({"direction": (0.0, 0.0, 1.0)}, {"intensity": -0.1}):
        with pytest.raises(ValueError):
            _sun(**changes)


def test_far_ground_is_lit_with_the_sun_and_dark_without() -> None:
    mujoco = pytest.importorskip("mujoco")

    def far_ground_mean(xml: str) -> float:
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        try:
            renderer = mujoco.Renderer(model, 120, 160)
        except (ImportError, RuntimeError, OSError) as exc:  # no GL context
            pytest.skip(f"no offscreen GL: {exc}")
        camera = mujoco.MjvCamera()
        camera.lookat[:] = (150.0, 0.0, 0.0)
        camera.distance, camera.azimuth, camera.elevation = 5.0, 0.0, -10.0
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        frame = renderer.render().astype(float)
        renderer.close()
        return float(frame[90:, :, 1].mean())  # lower rows: the far turf

    lit = far_ground_mean(add_sun(WORLD, load_sun_light()))
    assert lit > far_ground_mean(WORLD) + 20.0
