"""Scene lighting of the MuJoCo visual layer (#15299): settings and render quality.

The render tests measure ragged self-shadow (features narrower than a settings
window) and clipped skin as shares of the skin pixels, against settings bounds.
"""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from defusedxml import ElementTree as ET

from src.engines.physics_engines.mujoco.python import full_body_mjcf as exporter
from src.engines.physics_engines.mujoco.python import scene_lighting as lighting
from src.shared.python.model_appearance import document_from_dict
from src.shared.python.motion_matching.full_body_v2_showpiece import SHOWPIECE_V2
from src.tools.native_viewer_export.backends.mujoco_worker import build_scene_xml

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
FULL_BODY = ROOT / "docs/development/full_body_models/full_body_spec_v1.json"
RAW = json.loads(lighting.SETTINGS_PATH.read_text())


def _scene_xml() -> str:
    doc = document_from_dict({"schema_version": "appearance-v1"})
    return exporter.export_full_body_mjcf(FULL_BODY.read_bytes(), appearance=doc)[0]


def _numbers(text: str) -> list[float]:
    return [float(value) for value in text.split()]


def test_emitted_lights_and_quality_come_from_the_settings_file() -> None:
    settings = lighting.load_scene_lighting()
    root = ET.fromstring(_scene_xml())
    lights = {e.get("name"): e for e in root.iter("light")}
    assert list(lights) == [f"visual_{light.name}" for light in settings.studio]
    for light in settings.studio:
        element = lights[f"visual_{light.name}"]
        assert _numbers(element.get("diffuse")) == list(light.diffuse)
        assert _numbers(element.get("specular")) == list(light.specular)
        assert element.get("castshadow") == str(light.castshadow).lower()
    # Only the key light casts a shadow: a shadowed fill doubles every fringe.
    assert [light.name for light in settings.studio if light.castshadow] == ["key"]
    visual = root.find("visual")
    assert int(visual.find("quality").get("shadowsize")) == settings.shadowsize
    shadow_map = visual.find("map")
    assert float(shadow_map.get("shadowclip")) == settings.shadowclip
    assert float(shadow_map.get("shadowscale")) == settings.shadowscale
    assert float(shadow_map.get("zfar")) == settings.zfar
    head = settings.headlight["studio"]
    assert _numbers(visual.find("headlight").get("ambient")) == list(head["ambient"])


def test_light_position_follows_offset_height_and_ground_normal() -> None:
    world = lighting.ET.Element("worldbody")
    light = lighting.load_scene_lighting().studio[0]
    assert lighting.add_lights(world, np.array([0.0, 0.0, 1.0]), (light,)) == 1
    position = _numbers(world[0].get("pos"))
    assert position == [*light.offset_m, light.height_m]
    assert _numbers(world[0].get("dir")) == [-value for value in position]


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("quality", "shadowsize"), 3000, "power of two"),
        (("quality", "offsamples"), -1, "offsamples"),
        (("map", "shadowclip"), 0.0, "shadowclip"),
        (("map", "shadowscale"), 1.5, "shadowscale"),
        (("map", "zfar"), 0.01, "zfar"),
        (("map", "haze"), 2.0, "haze"),
        (("rigs", "studio", 0, "diffuse"), [0.5, 0.5, 1.2], "diffuse"),
        (("rigs", "studio", 0, "specular"), [0.5, 0.5], "specular"),
        (("rigs", "studio", 0, "height_m"), 0.0, "height_m"),
        (("rigs", "studio", 0, "castshadow"), "yes", "castshadow"),
        (("rigs", "studio"), [], "at least one light"),
        (("acceptance", "max_clipped_share"), 1.5, "max_clipped_share"),
    ],
)
def test_invalid_settings_are_rejected(tmp_path, path, value, match) -> None:
    raw = copy.deepcopy(RAW)
    node = raw
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    target = tmp_path / "scene_lighting.json"
    target.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match=match):
        lighting.load_scene_lighting(target)


def test_native_skin_mask_excludes_occluded_and_unchanged_pixels() -> None:
    with_skin = np.array([[1.0, 2.0], [3.0, 4.0]])
    without_skin = np.array([[2.0, 2.0], [2.0, 4.00000001]])
    assert _native_skin_mask(with_skin, without_skin).tolist() == [
        [True, False],
        [False, False],
    ]


def _native_skin_mask(with_skin: np.ndarray, without_skin: np.ndarray) -> np.ndarray:
    """Visible skin surfaces are closer than the scene with skins hidden."""
    return with_skin < without_skin - 1e-5


def _skin_mask(mujoco, model, segmentation) -> np.ndarray:
    skin = np.zeros(model.ngeom + 1, dtype=bool)
    for geom in range(model.ngeom):
        material = model.geom_matid[geom]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, material)
        skin[geom] = material >= 0 and (name or "").startswith("mat_skin")
    is_geom = segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    return is_geom & skin[np.clip(segmentation[..., 0], -1, model.ngeom - 1)]


def _measure(mujoco, model, bounds) -> dict[str, float]:
    """Ragged-shadow and clipped shares of the skin over the acceptance views."""
    ndimage = pytest.importorskip("scipy.ndimage")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    width, height = bounds["image_size_px"]
    try:
        renderer = mujoco.Renderer(model, height, width)
    except (RuntimeError, ValueError, OSError) as error:
        pytest.skip(f"no offscreen GL: {error}")
    window = int(bounds["ragged_window_px"])
    kernel: np.ndarray = np.ones((window, window), dtype=bool)
    totals = dict.fromkeys(("ragged", "inner", "clipped", "skin", "native"), 0)
    options = mujoco.MjvOption()
    original_groups = model.skin_group.copy()
    if model.nskin:
        body_skin = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SKIN, "body_skin_skin")
        assert body_skin >= 0, "native body skin is missing"
        model.skin_group[:] = 0
        model.skin_group[body_skin] = 5
        options.skingroup[5] = 1
    camera = mujoco.MjvCamera()
    scene = renderer.scene
    try:
        for azimuth, elevation, distance in bounds["views_azimuth_elevation_distance"]:
            camera.lookat[:] = data.xpos[1]
            camera.azimuth, camera.elevation = azimuth, elevation
            camera.distance = distance
            frames = []
            for shadow in (1, 0):
                renderer.update_scene(data, camera=camera, scene_option=options)
                scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = shadow
                frames.append(renderer.render().astype(float))
            renderer.enable_segmentation_rendering()
            renderer.update_scene(data, camera=camera, scene_option=options)
            mask = _skin_mask(mujoco, model, renderer.render())
            renderer.disable_segmentation_rendering()
            if model.nskin:
                renderer.enable_depth_rendering()
                renderer.update_scene(data, camera=camera, scene_option=options)
                with_skin = renderer.render().copy()
                options.skingroup[5] = 0
                renderer.update_scene(data, camera=camera, scene_option=options)
                without_skin = renderer.render().copy()
                options.skingroup[5] = 1
                renderer.disable_depth_rendering()
                native = _native_skin_mask(with_skin, without_skin)
                totals["native"] += int(native.sum())
                mask |= native
            lit, unshadowed = frames
            shadowed = (unshadowed - lit).mean(axis=2) > bounds["shadow_step_levels"]
            thin = shadowed & ~ndimage.binary_opening(shadowed, kernel)
            thin |= ndimage.binary_closing(shadowed, kernel) & ~shadowed
            inner = ndimage.binary_erosion(mask, kernel)
            totals["ragged"] += int((thin & inner).sum())
            totals["inner"] += int(inner.sum())
            totals["clipped"] += int(
                (mask & (lit.max(axis=2) >= bounds["clip_level"])).sum()
            )
            totals["skin"] += int(mask.sum())
    finally:
        model.skin_group[:] = original_groups
        renderer.close()
    if model.nskin:
        assert totals["native"] > 0, "native skinned body contributed no pixels"
    assert totals["inner"] > bounds["min_skin_px"]
    return {
        "ragged": totals["ragged"] / totals["inner"],
        "clipped": totals["clipped"] / totals["skin"],
    }


@pytest.mark.requires_gl
@pytest.mark.parametrize("sun", [False, True], ids=["spot", "sun"])
def test_lit_skin_has_no_ragged_shadow_and_is_not_clipped(sun: bool) -> None:
    """``sun``: lights made directional, as a wide outdoor render does."""
    mujoco = pytest.importorskip("mujoco")
    bounds = lighting.load_scene_lighting().acceptance
    model = mujoco.MjModel.from_xml_string(_scene_xml())
    if sun:
        model.light_type[:] = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    shares = _measure(mujoco, model, bounds)
    assert shares["ragged"] <= bounds["max_ragged_shadow_share"], shares
    assert shares["clipped"] <= bounds["max_clipped_share"], shares


_SHOWPIECE_LIGHTING_XFAIL = (
    "Refs #15383: showpiece native skin exceeds acceptance limits until lighting "
    "is visually qualified (spot ragged 0.01922 vs 0.007, clipped 0.02775 vs 0.03; "
    "directional ragged 0.01999, clipped 0.08999 vs 0.03)."
)


@pytest.mark.requires_gl
@pytest.mark.xfail(strict=False, reason=_SHOWPIECE_LIGHTING_XFAIL)
@pytest.mark.parametrize("sun", [False, True], ids=["spot", "sun"])
def test_showpiece_native_skin_meets_lighting_bounds(sun: bool) -> None:
    """Measure the actual native export, including its visible skinned body."""
    mujoco = pytest.importorskip("mujoco")
    spec = json.loads(SHOWPIECE_V2.read_text(encoding="utf-8"))
    spec["subject"].update(
        body_visual_backend="skinned_cc0", body_skin_profile="showpiece"
    )
    xml, _ = build_scene_xml(json.dumps(spec).encode())
    model = mujoco.MjModel.from_xml_string(xml)
    assert model.nskin > 0
    if sun:
        model.light_type[:] = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    bounds = lighting.load_scene_lighting().acceptance
    shares = _measure(mujoco, model, bounds)
    assert shares["ragged"] <= bounds["max_ragged_shadow_share"], shares
    assert shares["clipped"] <= bounds["max_clipped_share"], shares
