"""Skin load shading settings: impact blanking, load alias, kept colours (#15150)."""

import dataclasses
import json

import pytest

from src.shared.python.body_part_viz import AxialLoadFrame, ForceColorScale
from src.shared.python.body_part_viz import skin_load_settings as sls

pytestmark = pytest.mark.unit

SETTINGS = sls.load_skin_load_settings()
PELVIS, LUMBAR = next(iter(SETTINGS.load_body_alias.items()))
XML = """<mujoco><asset>
<material name="mat_skin_light" rgba="0.8 0.6 0.5 1"/>
<material name="mat_hair_brown" rgba="0.3 0.2 0.1 1"/>
<material name="mat_iris_hazel" rgba="0.4 0.3 0.1 1"/>
</asset><worldbody><body name="head">
<geom name="face" size="0.1" material="mat_skin_light"/>
<geom name="hair" size="0.1" pos="0 0 0.1" material="mat_hair_brown"/>
<geom name="iris" size="0.01" pos="0.1 0 0" material="mat_iris_hazel"/>
<geom name="bare" size="0.01" pos="0 0.1 0" rgba="0.5 0.5 0.5 1"/>
</body></worldbody></mujoco>"""


def test_settings_come_from_the_json_file_with_a_source_for_each():
    raw = json.loads(sls.SETTINGS_PATH.read_text(encoding="utf-8"))
    assert SETTINGS.axial_impact_blackout is raw["axial_impact_blackout"] is True
    assert SETTINGS.load_body_alias == raw["load_body_alias"]
    assert (
        list(SETTINGS.unshaded_material_prefixes) == (raw["unshaded_material_prefixes"])
    )
    assert set(raw["sources"]) == set(raw) - {"sources"}


def test_the_default_alias_gives_the_pelvis_the_lumbar_load():
    from src.shared.python.motion_matching.full_body_v2 import LUMBAR_BODY

    assert LUMBAR == LUMBAR_BODY and PELVIS.endswith("LowerTorso")


def test_invalid_settings_are_rejected():
    with pytest.raises(TypeError, match="true or false"):
        dataclasses.replace(SETTINGS, axial_impact_blackout=1)
    with pytest.raises(TypeError, match="load_body_alias"):
        dataclasses.replace(SETTINGS, load_body_alias={"pelvis": 3})
    with pytest.raises(ValueError, match="chain"):
        dataclasses.replace(SETTINGS, load_body_alias={"a": "b", "b": "c"})
    with pytest.raises(TypeError, match="unshaded_material_prefixes"):
        dataclasses.replace(SETTINGS, unshaded_material_prefixes="mat_hair")
    with pytest.raises(TypeError, match="unshaded_material_prefixes"):
        dataclasses.replace(SETTINGS, unshaded_material_prefixes=("",))


def test_an_aliased_body_takes_another_bodys_load_and_the_rest_keep_theirs():
    pieces = {"shorts": PELVIS, "thigh": "femur_l", "shirt_hem": PELVIS}
    alias = {**SETTINGS.load_body_alias, "not_in_the_model": "femur_r"}
    assert sls.aliased_load_bodies(pieces, alias) == {
        "shorts": LUMBAR,
        "thigh": "femur_l",
        "shirt_hem": LUMBAR,
    }
    assert pieces["shorts"] == PELVIS
    assert sls.aliased_load_bodies(pieces, {}) == pieces
    with pytest.raises(TypeError):
        sls.aliased_load_bodies(pieces, ["pelvis"])


def _scene():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    scene = mujoco.MjvScene(model, maxgeom=10)
    mujoco.mj_forward(model, data)
    mujoco.mjv_updateScene(
        model,
        data,
        mujoco.MjvOption(),
        None,
        mujoco.MjvCamera(),
        mujoco.mjtCatBit.mjCAT_ALL,
        scene,
    )
    return model, scene


def _colours(model, scene):
    return {
        model.geom(int(g.objid)).name: [float(c) for c in g.rgba]
        for g in scene.geoms[: scene.ngeom]
    }


def test_hair_and_eyes_keep_their_colour_while_the_face_is_shaded():
    from src.shared.python.body_part_viz import mujoco_force_colors as mfc

    model, scene = _scene()
    keep = mfc.unshaded_geom_ids(model, SETTINGS.unshaded_material_prefixes)
    assert keep == {model.geom("hair").id, model.geom("iris").id}
    assert mfc.unshaded_geom_ids(model, ()) == frozenset()
    before = _colours(model, scene)
    frame = AxialLoadFrame(0.0, {"head": -5000.0}, "test")
    scale = ForceColorScale(enabled=True)
    mfc.apply_mujoco_scene_colors(model, scene, frame, scale, keep_geoms=keep)
    after = _colours(model, scene)
    assert after["hair"] == before["hair"] and after["iris"] == before["iris"]
    assert after["face"] != before["face"] and after["bare"] != before["bare"]
    mfc.apply_mujoco_scene_colors(model, scene, frame, scale)
    assert _colours(model, scene)["hair"] != before["hair"]  # the default: all shaded
    with pytest.raises(TypeError, match="prefixes"):
        mfc.unshaded_geom_ids(model, "mat_hair")
