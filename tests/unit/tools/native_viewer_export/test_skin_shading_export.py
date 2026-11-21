"""Axial-load shading of a skinned body in the MuJoCo export path (#15150)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from defusedxml import ElementTree as ET

from src.shared.python.body_part_viz.axial_loads import AxialLoadFrame
from src.shared.python.body_part_viz.force_colors import ForceColorScale
from src.shared.python.body_part_viz.skin_load_settings import (
    SkinLoadSettings,
    load_skin_load_settings,
)
from src.tools.native_viewer_export.backends import _skin_shading as shading
from src.tools.native_viewer_export.backends._scene_overlays import (
    apply_scene_overlays,
)

mujoco = pytest.importorskip("mujoco")
pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

ROOT = Path(__file__).resolve().parents[4]
SPEC_V2 = (
    ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver_v2.json"
)
SCALE = ForceColorScale(
    enabled=True, opacity=1.0, tension_limit_n=100.0, compression_limit_n=100.0
)
CLOTH, TONE = (0.2, 0.4, 0.6), 0.5
SETTINGS = SkinLoadSettings(True, {"near": "far", "nobody": "ghost"}, ("mat_hair",))


def _xml(skin: bool = True) -> str:
    """Two bodies; on demand a four-quad skin strip blended from one to the other."""
    grid = [(float(i), float(j), 0.0) for i in range(5) for j in range(2)]
    faces = [f for i in range(4) for f in ((2 * i, 2 * i + 2, 2 * i + 1),)]
    faces += [(2 * i + 2, 2 * i + 3, 2 * i + 1) for i in range(4)]
    ids = " ".join(str(i) for i in range(len(grid)))
    far = [x / 4.0 for x, _, _ in grid]
    bones = "".join(
        f'<bone body="{body}" bindpos="0 0 0" bindquat="1 0 0 0" vertid="{ids}"'
        f' vertweight="{" ".join(format(w, ".6g") for w in share)}"/>'
        for body, share in (("near", [1.0 - w for w in far]), ("far", far))
    )
    deformable = (
        '<deformable><skin name="strip" material="cloth" vertex="'
        + " ".join(format(v, "g") for row in grid for v in row)
        + '" face="'
        + " ".join(str(i) for face in faces for i in face)
        + f'">{bones}</skin></deformable>'
    )
    return (
        "<mujoco><asset>"
        f'<texture name="weave" type="cube" builtin="flat" rgb1="{TONE} {TONE} {TONE}"'
        ' width="4" height="4"/>'
        '<material name="cloth" rgba="0.2 0.4 0.6 1" texture="weave"/>'
        '<material name="mat_hair_brown" rgba="0.3 0.2 0.1 1"/>'
        "</asset><worldbody>"
        '<body name="near"><joint/><geom name="limb" size="0.1" rgba=".5 .5 .5 1"/>'
        '<geom name="hair" size="0.1" pos="0 0 1" material="mat_hair_brown"/>'
        '<body name="far" pos="4 0 0"><joint/><geom size="0.1"/></body></body>'
        f"</worldbody>{deformable if skin else ''}</mujoco>"
    )


def _frames(**loads: float | None) -> list[AxialLoadFrame | None]:
    return [AxialLoadFrame(0.0, dict(loads), "test")]


def _material_rgb(model: Any, name: str) -> tuple[float, ...]:
    return tuple(float(c) for c in model.mat_rgba[model.material(name).id][:3])


def _scene(model: Any) -> Any:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    scene = mujoco.MjvScene(model, maxgeom=model.ngeom + 50)
    mujoco.mjv_updateScene(
        model,
        data,
        mujoco.MjvOption(),
        None,
        mujoco.MjvCamera(),
        mujoco.mjtCatBit.mjCAT_ALL,
        scene,
    )
    return scene


def _geom_rgb(model: Any, scene: Any, name: str) -> tuple[float, ...]:
    wanted = model.geom(name).id
    for index in range(scene.ngeom):
        geom = scene.geoms[index]
        if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and geom.objid == wanted:
            return tuple(float(c) for c in geom.rgba[:3])
    raise AssertionError(f"geom {name!r} is not in the scene")


@pytest.mark.parametrize(
    ("frames", "scale"),
    [
        (None, SCALE),
        (_frames(near=1.0), None),
        (_frames(near=1.0), ForceColorScale(enabled=False)),
        (_frames(near=1.0), ForceColorScale(enabled=True, opacity=0.0)),
    ],
)
def test_the_split_is_wanted_only_while_shading_is_on(frames: Any, scale: Any) -> None:
    assert shading.skin_pieces_wanted(frames, scale) is False
    assert shading.skin_pieces_wanted(_frames(near=1.0), SCALE) is True


def test_scene_xml_is_the_same_object_when_off_or_without_a_skin() -> None:
    skinned, plain = _xml(), _xml(skin=False)
    for xml, wanted in ((skinned, False), (plain, True)):
        same, table = shading.split_scene_skins(xml, wanted)
        assert same is xml and table == {}


def test_model_without_pieces_binds_nothing_and_keeps_its_materials() -> None:
    model, bound = shading.compile_shaded_model(_xml(), None, SCALE, SETTINGS)
    reference = mujoco.MjModel.from_xml_string(_xml())
    assert bound == shading.SkinShading()
    assert bound.keep_geoms == frozenset() and model.nskin == reference.nskin == 1
    assert np.array_equal(model.mat_rgba, reference.mat_rgba)
    assert bound.colour(model, _frames(near=50.0)[0], SCALE) == 0
    assert np.array_equal(model.mat_rgba, reference.mat_rgba)


def test_overlay_step_without_kept_geoms_shades_as_before() -> None:
    model = mujoco.MjModel.from_xml_string(_xml(skin=False))
    frames, old, new = _frames(near=100.0), _scene(model), _scene(model)
    apply_scene_overlays(model, old, 0, frames, SCALE, None)
    apply_scene_overlays(model, new, 0, frames, SCALE, None, keep_geoms=frozenset())
    for index in range(old.ngeom):
        assert np.array_equal(old.geoms[index].rgba, new.geoms[index].rgba)
    assert _geom_rgb(model, new, "hair") == pytest.approx((0.0, 0.0, 1.0))


def test_skin_is_cut_bound_toned_and_follows_the_frame_load() -> None:
    frames = _frames(far=100.0)
    model, bound = shading.compile_shaded_model(_xml(), frames, SCALE, SETTINGS)
    assert model.nskin == 2 and len(bound.pieces) == 2
    # the alias moves the near piece to the far body's load; an alias to a
    # body the model lacks is left out instead of failing the export
    assert {piece.body for piece in bound.pieces} == {"far"}
    expected = tuple(c * TONE for c in CLOTH)  # the texture is stored in 8 bits
    for piece in bound.pieces:
        assert piece.base_rgba[:3] == pytest.approx(expected, abs=1 / 255)
        assert _material_rgb(model, piece.material) == piece.base_rgba[:3]
    assert _material_rgb(model, "cloth") == pytest.approx(CLOTH, abs=1e-6)
    assert bound.colour(model, frames[0], SCALE) == 2
    for piece in bound.pieces:
        assert _material_rgb(model, piece.material) == pytest.approx((0.0, 0.0, 1.0))
    assert bound.colour(model, None, SCALE) == 0  # a material keeps its last colour
    for piece in bound.pieces:
        assert _material_rgb(model, piece.material) == piece.base_rgba[:3]


def test_kept_materials_keep_their_colour_under_scene_shading() -> None:
    frames = _frames(near=100.0, far=100.0)
    model, bound = shading.compile_shaded_model(_xml(), frames, SCALE, SETTINGS)
    assert bound.keep_geoms == frozenset({model.geom("hair").id})
    scene = _scene(model)
    apply_scene_overlays(
        model, scene, 0, frames, SCALE, None, keep_geoms=bound.keep_geoms
    )
    assert _geom_rgb(model, scene, "hair") == pytest.approx((0.3, 0.2, 0.1))
    assert _geom_rgb(model, scene, "limb") == pytest.approx((0.0, 0.0, 1.0))


def test_bad_inputs_are_rejected() -> None:
    model = mujoco.MjModel.from_xml_string(_xml())
    with pytest.raises(TypeError):
        shading.split_scene_skins(b"<mujoco/>", True)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        shading.skin_pieces_wanted(_frames(near=1.0), "on")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        shading.bind_skin_shading(object(), {}, SETTINGS)
    with pytest.raises(TypeError):
        shading.bind_skin_shading(model, {"strip": "near"}, "x")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown skin"):
        shading.bind_skin_shading(model, {"no_such_piece": "near"}, SETTINGS)
    with pytest.raises(ValueError, match="material"):
        shading.texture_tone(model, model.nmat)


@pytest.fixture(scope="module")
def golfer() -> tuple[Any, shading.SkinShading, list[AxialLoadFrame | None]]:
    from src.tools.native_viewer_export.backends.mujoco_worker import build_scene_xml

    spec = json.loads(SPEC_V2.read_text(encoding="utf-8"))
    spec["subject"]["body_visual_backend"] = "skinned_cc0"
    xml, _ = build_scene_xml(json.dumps(spec).encode())
    bodies = [b.get("name") for b in ET.fromstring(xml).iter("body")]
    frames = _frames(**dict.fromkeys(bodies, -100.0))
    model, bound = shading.compile_shaded_model(xml, frames, SCALE)
    return model, bound, frames


def test_golfer_skin_pieces_all_take_a_load_colour(golfer: Any) -> None:
    model, bound, frames = golfer
    alias = load_skin_load_settings().load_body_alias
    assert model.nskin == len(bound.pieces) > 3  # more pieces than layers
    loads = {piece.body for piece in bound.pieces}
    assert set(alias.values()) <= loads and not set(alias) & loads
    assert bound.colour(model, frames[0], SCALE) == len(bound.pieces)
    for piece in bound.pieces:
        assert _material_rgb(model, piece.material) == pytest.approx((1.0, 0.0, 0.0))
    bound.colour(model, None, SCALE)


def test_golfer_hair_eyes_brows_and_lips_keep_their_colour(golfer: Any) -> None:
    model, bound, frames = golfer
    prefixes = load_skin_load_settings().unshaded_material_prefixes
    kept = {model.material(int(model.geom_matid[g])).name for g in bound.keep_geoms}
    for prefix in ("mat_hair", "mat_eye", "mat_brow", "mat_lip"):
        assert any(name.startswith(prefix) for name in kept), prefix
    assert all(name.startswith(prefixes) for name in kept)
    plain, shaded = _scene(model), _scene(model)
    apply_scene_overlays(
        model, shaded, 0, frames, SCALE, None, keep_geoms=bound.keep_geoms
    )
    same = [
        np.array_equal(shaded.geoms[i].rgba, plain.geoms[i].rgba)
        for i in range(plain.ngeom)
    ]
    for index, geom in enumerate(shaded.geoms[: plain.ngeom]):
        if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and geom.objid in bound.keep_geoms:
            assert same[index]
    assert same.count(False) > len(bound.keep_geoms)
