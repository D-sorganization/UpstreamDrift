"""MuJoCo visual layer: capsule skeleton, floor and lights without physics change."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python import full_body_mjcf as exporter

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
FULL_BODY = ROOT / "docs/development/full_body_models/full_body_spec_v1.json"
# full_body_spec_v1.json predates the Grip-solid naming club_assembly needs
# (OSV-8, #11755), so the decorative-ball tests use a spec where the club
# actually resolves.
CLUB_BODY = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"


def test_visual_export_adds_geoms_but_keeps_physics_identical() -> None:
    mujoco = pytest.importorskip("mujoco")
    raw = FULL_BODY.read_bytes()
    plain_xml, plain_meta = exporter.export_full_body_mjcf(raw)
    visual_xml, visual_meta = exporter.export_full_body_mjcf(raw, visual=True)
    plain = mujoco.MjModel.from_xml_string(plain_xml)
    visual = mujoco.MjModel.from_xml_string(visual_xml)
    assert visual.nq == plain.nq == 41 and visual.nv == plain.nv == 41
    assert visual.nbody == plain.nbody
    assert visual.ngeom > plain.ngeom + 20  # one capsule per body plus floor
    assert visual.nlight >= 1
    assert visual_meta["visual_layer"]["capsules"] >= plain.nbody - 1
    assert visual_meta["visual_layer"]["floor"] is True
    assert "visual_layer" not in plain_meta
    # Visual geoms never collide and never contribute inertia.
    for i in range(visual.ngeom):
        if visual.geom_group[i] == 1:
            assert visual.geom_contype[i] == 0 and visual.geom_conaffinity[i] == 0
    np.testing.assert_allclose(visual.body_mass, plain.body_mass)
    np.testing.assert_allclose(visual.body_inertia, plain.body_inertia)
    d_plain, d_visual = mujoco.MjData(plain), mujoco.MjData(visual)
    rng = np.random.default_rng(0)
    q = rng.normal(scale=0.2, size=plain.nq)
    d_plain.qpos[:] = q
    d_visual.qpos[:] = q
    mujoco.mj_forward(plain, d_plain)
    mujoco.mj_forward(visual, d_visual)
    np.testing.assert_allclose(d_visual.xpos, d_plain.xpos, atol=1e-12)
    np.testing.assert_allclose(d_visual.qacc, d_plain.qacc, atol=1e-9)


@pytest.mark.requires_gl
def test_visual_export_renders_offscreen(tmp_path: Path) -> None:
    mujoco = pytest.importorskip("mujoco")
    xml, _ = exporter.export_full_body_mjcf(FULL_BODY.read_bytes(), visual=True)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    try:
        renderer = mujoco.Renderer(model, 120, 160)
    except (RuntimeError, ValueError, OSError) as error:  # headless CI without GL
        pytest.skip(f"no offscreen GL: {error}")
    renderer.update_scene(data)
    image = renderer.render()
    assert image.shape == (120, 160, 3)
    assert image.mean() > 5.0  # not a black frame: floor and lit capsules visible


def _ball_geom_id(mujoco, model) -> int:  # noqa: ANN001
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "visual_ball")
    assert gid >= 0, "visual_ball geom missing from the exported model"
    return gid


def _club_body_and_offsets():  # noqa: ANN202
    """The real driver club assembly plus the exporter's own reference offsets."""
    import xml.etree.ElementTree as ET  # nosec B405 # nosemgrep: python.lang.security.use-defused-xml.use-defused-xml - construction only; parsing is defused

    from src.engines.physics_engines.mujoco.python import full_body_mjcf as exp
    from src.shared.python.model_appearance import club_assembly as ca

    spec = json.loads(CLUB_BODY.read_bytes())
    club = ca.assembly_from_spec(spec)
    club_body = ca.club_body_name(spec)
    _, offsets = exp._build_full_body_kinematics(ET.Element("mujoco"), spec)
    return club, club_body, offsets


def test_club_face_world_transforms_club_frame_into_world() -> None:
    from src.engines.physics_engines.mujoco.python import visual_layer
    from src.shared.python.model_appearance import club_assembly as ca

    club, club_body, offsets = _club_body_and_offsets()
    centre, normal = visual_layer._club_face_world(club, club_body, offsets)
    offset = offsets[club_body]
    np.testing.assert_allclose(
        centre, offset[:3, :3] @ ca.clubface_centre(club) + offset[:3, 3]
    )
    np.testing.assert_allclose(normal, offset[:3, :3] @ ca.clubface_vector(club))


def test_attach_decorative_ball_computes_grounded_address_geometry() -> None:
    """A club body coincident with the world frame is a grounded reference pose."""
    import xml.etree.ElementTree as ET  # nosec B405 # nosemgrep: python.lang.security.use-defused-xml.use-defused-xml - construction only; parsing is defused

    from src.engines.physics_engines.mujoco.python import visual_layer
    from src.shared.python.model_appearance import club_assembly as ca
    from src.shared.python.model_appearance.ball import (
        BALL_RADIUS_M,
        ball_position_at_address,
    )
    from src.shared.python.model_appearance.schema import BallSettings

    club, club_body, _ = _club_body_and_offsets()
    world = ET.Element("worldbody")
    elements = {"world": world, club_body: ET.SubElement(world, "body")}
    offsets = {club_body: np.eye(4)}
    meta = visual_layer._attach_decorative_ball(
        elements, offsets, club, club_body, BallSettings(), 0.0
    )
    assert meta["enabled"] is True and meta["source"] == "address_geometry"
    position = np.asarray(meta["position_m"])
    assert position[2] == pytest.approx(BALL_RADIUS_M, abs=1e-9)  # on the ground
    # Matches the shared address-geometry formula exactly (face centre offset
    # by one radius along the face normal, then rested on the ground).
    expected = ball_position_at_address(
        ca.clubface_centre(club), ca.clubface_vector(club), ground_height_m=0.0
    )
    np.testing.assert_allclose(position, expected, atol=1e-12)
    ball_geom = world.find("geom")
    assert ball_geom is not None
    assert ball_geom.get("name") == "visual_ball"
    assert ball_geom.get("class") == "visual"


def test_attach_decorative_ball_is_unavailable_when_reference_pose_is_ungrounded() -> (
    None
):
    """The real spec's static reference pose is a rest stance, not an address
    stance (its clubhead sits roughly a metre off the ground) — the ball must
    be reported unavailable with a reason rather than drawn somewhere wrong.
    """
    from src.engines.physics_engines.mujoco.python import visual_layer
    from src.shared.python.model_appearance.schema import BallSettings

    club, club_body, offsets = _club_body_and_offsets()
    elements = {"world": object(), club_body: object()}
    meta = visual_layer._attach_decorative_ball(
        elements, offsets, club, club_body, BallSettings(), 0.0
    )
    assert meta["enabled"] is False
    assert "not grounded" in meta["reason"] or "reference pose" in meta["reason"]


def test_attach_decorative_ball_override_position_bypasses_groundedness_check() -> None:
    """An explicit ``position_m`` (e.g. a measured ball) is trusted verbatim,
    even though this spec's reference pose is not itself grounded.
    """
    import xml.etree.ElementTree as ET  # nosec B405 # nosemgrep: python.lang.security.use-defused-xml.use-defused-xml - construction only; parsing is defused

    from src.engines.physics_engines.mujoco.python import visual_layer
    from src.shared.python.model_appearance.schema import BallSettings

    club, club_body, offsets = _club_body_and_offsets()
    world = ET.Element("worldbody")
    elements = {"world": world, club_body: ET.SubElement(world, "body")}
    measured = (1.0, 2.0, 0.021335)
    meta = visual_layer._attach_decorative_ball(
        elements,
        offsets,
        club,
        club_body,
        BallSettings(True, measured, "measured"),
        0.0,
    )
    assert meta == {"enabled": True, "position_m": list(measured), "source": "measured"}


def test_attach_decorative_ball_disabled_is_reported_with_a_reason() -> None:
    from src.engines.physics_engines.mujoco.python import visual_layer
    from src.shared.python.model_appearance.schema import BallSettings

    club, club_body, offsets = _club_body_and_offsets()
    elements = {"world": object(), club_body: object()}
    meta = visual_layer._attach_decorative_ball(
        elements, offsets, club, club_body, BallSettings(enabled=False), 0.0
    )
    assert meta == {"enabled": False, "reason": "ball.enabled is False"}


def test_attach_decorative_ball_reports_missing_club() -> None:
    from src.engines.physics_engines.mujoco.python import visual_layer
    from src.shared.python.model_appearance.schema import BallSettings

    meta = visual_layer._attach_decorative_ball(
        {"world": object()}, {}, None, None, BallSettings(), 0.0
    )
    assert meta == {"enabled": False, "reason": "spec has no club body"}


def test_decorative_ball_appears_in_exported_mjcf_with_an_explicit_position() -> None:
    mujoco = pytest.importorskip("mujoco")
    from src.shared.python.model_appearance.schema import (
        AppearanceDocument,
        BallSettings,
        document_from_dict,
    )

    raw = CLUB_BODY.read_bytes()
    doc_off = document_from_dict(
        {"schema_version": "appearance-v1", "ball": {"enabled": False}}
    )
    xml, meta = exporter.export_full_body_mjcf(raw, appearance=doc_off)
    model = mujoco.MjModel.from_xml_string(xml)
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "visual_ball") == -1
    assert meta["visual_layer"]["ball"]["enabled"] is False

    measured = (1.0, 2.0, 0.021335)
    doc_override = AppearanceDocument(ball=BallSettings(True, measured, "measured"))
    xml2, meta2 = exporter.export_full_body_mjcf(raw, appearance=doc_override)
    model2 = mujoco.MjModel.from_xml_string(xml2)
    gid2 = _ball_geom_id(mujoco, model2)
    np.testing.assert_allclose(model2.geom_pos[gid2], measured, atol=1e-9)
    assert meta2["visual_layer"]["ball"]["source"] == "measured"
    assert model2.geom_contype[gid2] == 0 and model2.geom_conaffinity[gid2] == 0
    assert model2.geom_group[gid2] == 1 and model2.geom_bodyid[gid2] == 0  # world body


def test_decorative_ball_is_fixed_to_world_and_never_moves() -> None:
    mujoco = pytest.importorskip("mujoco")
    from src.shared.python.model_appearance.schema import (
        AppearanceDocument,
        BallSettings,
    )

    raw = CLUB_BODY.read_bytes()
    doc = AppearanceDocument(ball=BallSettings(True, (1.0, 2.0, 0.021335), "measured"))
    xml, _ = exporter.export_full_body_mjcf(raw, appearance=doc)
    model = mujoco.MjModel.from_xml_string(xml)
    gid = _ball_geom_id(mujoco, model)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    before = np.array(data.geom_xpos[gid])
    rng = np.random.default_rng(3)
    data.qpos[:] = rng.normal(scale=0.2, size=model.nq)
    mujoco.mj_forward(model, data)
    after = np.array(data.geom_xpos[gid])
    np.testing.assert_allclose(before, after, atol=1e-12)


def test_whole_body_com_and_scene_markers(tmp_path: Path) -> None:
    import mujoco

    from src.engines.physics_engines.mujoco.python import visual_layer

    xml, _ = exporter.export_full_body_mjcf(FULL_BODY.read_bytes(), visual=True)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    com = visual_layer.whole_body_com(model, data)
    masses = model.body_mass[1:]
    expected = (masses[:, None] * data.xipos[1:]).sum(axis=0) / masses.sum()
    np.testing.assert_allclose(com, expected, atol=1e-9)
    scene = mujoco.MjvScene(model, maxgeom=4000)
    option = mujoco.MjvOption()
    mujoco.mjv_updateScene(
        model, data, option, None, mujoco.MjvCamera(), mujoco.mjtCatBit.mjCAT_ALL, scene
    )
    before = scene.ngeom
    visual_layer.add_com_markers(scene, model, data, 0.0)
    assert scene.ngeom == before + 2
    with pytest.raises(ValueError):
        visual_layer.add_scene_marker(scene, com, -0.01, (1, 0, 0, 1))
