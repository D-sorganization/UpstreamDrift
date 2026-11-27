"""Opt-in showpiece lighting of the MuJoCo export is display only (#15150)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from defusedxml import ElementTree as ET
import numpy as np
import pytest

from src.tools.native_viewer_export.backends import _scene_lighting as lighting
from src.tools.native_viewer_export.backends._subprocess import stage_job
from src.tools.native_viewer_export.backends._sun_light import add_sun, load_sun_light
from src.tools.native_viewer_export.backends._worker_job import WorkerJob
from src.tools.native_viewer_export.backends.mujoco_worker import build_scene_xml
from src.tools.native_viewer_export.core import ExportSettings
from tests.unit.tools.native_viewer_export import test_appearance_flags as flags

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

#: compiled fields that hold lights, materials, textures or render settings,
#: and the name table, buffer byte size and size hash that follow the number of
#: lights; every other field is compared
DISPLAY_PREFIXES = ("light_", "mat_", "tex_", "name", "nlight", "nnames")
DISPLAY_PREFIXES += ("nbuffer", "signature")
DISPLAY_STRUCTS = ("vis", "stat")


GREENS = ((0.2, 0.4, 0.2), (0.4, 0.6, 0.3))
WORLD = (
    '<mujoco><asset><texture name="t" type="2d" builtin="checker" width="8" '
    'height="8" rgb1="{} {} {}" rgb2="{} {} {}"/><material name="m" texture="t"/>'
    '</asset><worldbody><light name="visual_key"/><geom name="visual_floor" '
    'type="plane" size="1 1 1" material="m"/></worldbody></mujoco>'
).format(*GREENS[0], *GREENS[1])


def _raw() -> dict[str, Any]:
    return json.loads(lighting.SETTINGS_PATH.read_text(encoding="utf-8"))


def _job(work: Path, *arguments: str) -> WorkerJob:
    work.mkdir()
    job = stage_job(flags._swing(), flags._settings(*arguments), [0], None, work)
    job.dump(work / "job.json")
    return WorkerJob.load(work / "job.json")


@pytest.fixture(scope="module")
def scene_xml() -> str:
    return build_scene_xml(flags.SPEC.read_bytes())[0]


def test_default_is_the_studio_scene_with_the_sun(tmp_path: Path) -> None:
    assert ExportSettings().lighting is None
    assert flags._settings("--lighting", "studio").lighting is None
    assert _job(tmp_path / "w").lighting is None
    assert lighting.light_scene(WORLD, None) == add_sun(WORLD, load_sun_light())


def test_flags_select_the_settings_and_the_job_carries_them(tmp_path: Path) -> None:
    chosen = flags._settings("--lighting", "showpiece").lighting
    assert chosen == lighting.load_lighting()
    raw = _raw()
    assert set(raw["sources"]) == set(raw) - {"sources"}
    raw["ground"]["texture_gain"] = 1.0
    custom = tmp_path / "custom.json"
    custom.write_text(json.dumps(raw), encoding="utf-8")
    other = flags._settings("--lighting-settings", str(custom)).lighting
    assert other.texture_gain == 1.0 and other != chosen
    carried = _job(tmp_path / "w", "--lighting", "showpiece").lighting
    assert lighting.LightingSettings.from_dict(carried) == chosen


def test_showpiece_lights_replace_the_scene_lights(scene_xml: str) -> None:
    mujoco = pytest.importorskip("mujoco")
    settings = lighting.load_lighting()
    lit = lighting.light_scene(scene_xml, settings.to_dict())
    model = mujoco.MjModel.from_xml_string(lit)
    assert model.nlight == len(settings.lights)
    directional = int(mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    for light in settings.lights:
        compiled = model.light(light.name)
        assert int(model.light_type[compiled.id]) == directional
        direction = np.array(light.direction)
        assert np.allclose(compiled.dir, direction / np.linalg.norm(direction))
        assert np.allclose(compiled.diffuse, light.diffuse)
        assert np.allclose(compiled.specular, light.specular)
        assert bool(compiled.castshadow[0]) is light.castshadow
    for key, value in settings.headlight.items():
        assert np.allclose(getattr(model.vis.headlight, key), value)
    assert model.vis.quality.shadowsize == settings.shadowsize
    assert model.vis.map.shadowclip == pytest.approx(settings.shadowclip)
    assert model.stat.extent == pytest.approx(settings.extent_m)
    floor = model.geom_matid[model.geom("visual_floor").id]
    assert model.mat_reflectance[floor] == pytest.approx(settings.reflectance)


def _unit(vector: Any) -> np.ndarray:
    return np.array(vector) / np.linalg.norm(vector)


def test_showpiece_key_shapes_the_body_without_clipping_it() -> None:
    """The look's invariants: a key above the ambient term and one short shadow."""
    settings = lighting.load_lighting()
    casters = [light for light in settings.lights if light.castshadow]
    ambient = max(settings.headlight["ambient"])
    # the body takes its form from the key, not from the flat ambient term
    assert sum(max(light.diffuse) for light in casters) > ambient
    # no cast shadow longer than the golfer is tall, and the copies of the key
    # close enough that the far end of the shadow is one edge, not steps
    directions = [_unit(light.direction) for light in casters]
    mean = _unit(np.mean(directions, axis=0))
    for direction in directions:
        assert np.degrees(np.arcsin(-direction[2])) >= 45.0
        assert np.degrees(np.arccos(min(1.0, direction @ mean))) <= 0.75
    # the brightest any surface can be lit: a white shirt and light skin clip
    # above about 1.1 times their base colour
    normals = np.random.default_rng(0).normal(size=(20000, 3))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    lit = np.full(len(normals), ambient + max(settings.headlight["diffuse"]))
    for light in settings.lights:
        facing = np.clip(-(normals @ _unit(light.direction)), 0.0, None)
        lit += max(light.diffuse) * facing
    assert lit.max() <= 1.1
    # the ground texture keeps both checker colours below the clip
    tex = ET.fromstring(lighting.apply_lighting(WORLD, settings)).find(".//texture")
    scale = settings.texture_gain * max(GREENS[0][1], GREENS[1][1])
    assert scale > 1.0 and max(float(v) for v in tex.get("rgb2").split()) <= 1.0


def test_ground_texture_takes_the_gain_and_contrast() -> None:
    settings = lighting.load_lighting()
    tex = ET.fromstring(lighting.apply_lighting(WORLD, settings)).find(".//texture")
    after = [[float(v) for v in tex.get(key).split()] for key in ("rgb1", "rgb2")]
    before = np.array(GREENS)
    mean = before.mean(axis=0)
    wanted = (
        mean + (before - mean) * settings.checker_contrast
    ) * settings.texture_gain
    assert np.allclose(after, np.clip(wanted, 0.0, 1.0), atol=1e-5)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("lights",), [], "at least one light"),
        (("lights", 0, "direction"), [0.0, 0.0, 1.0], "point down"),
        (("lights", 0, "diffuse"), [0.2, 1.2, 0.2], r"in \[0, 1\]"),
        (("lights", 0, "castshadow"), 1, "true or false"),
        (("lights", 1, "name"), "key_1", "unique"),
        (("headlight",), {"ambient": [0.1, 0.1, 0.1]}, "headlight"),
        (("shadow", "shadowsize"), 1000, "power of two"),
        (("shadow", "shadowclip"), 0.0, "shadowclip"),
        (("shadow", "extent_m"), -1.0, "extent_m"),
        (("ground", "texture_gain"), 0.0, "texture_gain"),
        (("ground", "checker_contrast"), 1.5, "checker_contrast"),
        (("ground", "reflectance"), 2.0, "reflectance"),
    ],
)
def test_invalid_settings_are_named(path: tuple, value: object, message: str) -> None:
    raw = _raw()
    node = raw
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        lighting.LightingSettings.from_dict(raw)
    with pytest.raises(ValueError, match="incomplete: 'shadowsize'"):
        lighting.LightingSettings.from_dict({**_raw(), "shadow": {}})
    with pytest.raises(ValueError, match="worldbody"):
        lighting.apply_lighting("<mujoco/>", lighting.load_lighting())


def test_lighting_leaves_every_physics_field_of_the_model(scene_xml: str) -> None:
    """Mass, inertia, geometry, contact and joint fields match with it on and off."""
    mujoco = pytest.importorskip("mujoco")
    studio = mujoco.MjModel.from_xml_string(lighting.light_scene(scene_xml, None))
    lit_xml = lighting.light_scene(scene_xml, lighting.load_lighting().to_dict())
    lit = mujoco.MjModel.from_xml_string(lit_xml)
    compared = []
    for name in dir(studio):
        if name.startswith(("_", *DISPLAY_PREFIXES)) or name in DISPLAY_STRUCTS:
            continue
        value = getattr(studio, name)
        if isinstance(value, np.ndarray):
            assert np.array_equal(value, getattr(lit, name)), name
        elif isinstance(value, (int, float)):
            assert value == getattr(lit, name), name
        else:
            continue
        compared.append(name)
    physics = ("body_mass", "body_inertia", "geom_size", "geom_pos", "geom_friction")
    physics += ("geom_solref", "jnt_axis", "jnt_range", "dof_damping", "nq", "nv")
    assert set(physics) <= set(compared)
    for field in ("timestep", "gravity", "integrator", "cone", "impratio"):
        assert np.array_equal(getattr(studio.opt, field), getattr(lit.opt, field))
    for field in ("meaninertia", "meanmass", "meansize", "center"):
        assert np.array_equal(getattr(studio.stat, field), getattr(lit.stat, field))
