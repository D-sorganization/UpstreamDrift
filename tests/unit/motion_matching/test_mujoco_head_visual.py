"""MuJoCo visual head: physics identity, placement, mocap gaze channel."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python import full_body_mjcf as exporter
from src.engines.physics_engines.mujoco.python.head_visual import drive_visual_head
from src.shared.python.model_appearance import document_from_dict

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
BODIES = ROOT / "docs/development/full_body_models"
DRIVER = BODIES / "full_body_spec_anthro_driver.json"
HEAD = "GolfSwing3D_Kinetic/Head"


def _export(spec: Path, head: dict | None, **kw):
    doc = {"schema_version": "appearance-v1"}
    if head is not None:
        doc["head"] = head
    return exporter.export_full_body_mjcf(
        spec.read_bytes(), appearance=document_from_dict(doc), **kw
    )


def _state(mujoco, model, seed=3):
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    data.qpos[:] = rng.normal(scale=0.05, size=model.nq)
    data.qvel[:] = rng.normal(scale=0.2, size=model.nv)
    mujoco.mj_forward(model, data)
    return data


def test_physics_identical_with_and_without_head() -> None:
    mujoco = pytest.importorskip("mujoco")
    with_head = mujoco.MjModel.from_xml_string(_export(DRIVER, {"enabled": True})[0])
    without = mujoco.MjModel.from_xml_string(_export(DRIVER, {"enabled": False})[0])
    plain = mujoco.MjModel.from_xml_string(
        exporter.export_full_body_mjcf(DRIVER.read_bytes())[0]
    )
    for other in (with_head, without):
        assert (other.nbody, other.nq, other.nv) == (plain.nbody, plain.nq, plain.nv)
        for attr in (
            "body_mass",
            "body_inertia",
            "body_ipos",
            "body_iquat",
            "body_subtreemass",
        ):
            np.testing.assert_array_equal(getattr(other, attr), getattr(plain, attr))
        np.testing.assert_array_equal(
            _state(mujoco, other).qacc, _state(mujoco, plain).qacc
        )


def test_head_geoms_ride_the_head_body_and_are_visual_only() -> None:
    mujoco = pytest.importorskip("mujoco")
    xml, meta = _export(DRIVER, {"headwear": "cap"})
    model = mujoco.MjModel.from_xml_string(xml)
    head = meta["visual_layer"]["head"]
    assert head["source"] == "head_body" and head["mode"] == "body"
    body_id = model.body(HEAD).id
    ids = [
        i
        for i in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith(
            "visual_head_"
        )
    ]
    assert len(ids) == len(head["parts"]) >= 12
    for i in ids:
        assert model.geom_bodyid[i] == body_id
        assert model.geom_contype[i] == 0 and model.geom_conaffinity[i] == 0
        assert model.geom_group[i] == 1


def test_head_placement_and_stature() -> None:
    mujoco = pytest.importorskip("mujoco")
    spec = json.loads(DRIVER.read_text())
    model = mujoco.MjModel.from_xml_string(_export(DRIVER, None)[0])
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body = model.body(HEAD)
    neck = np.array(data.xpos[body.id])
    head_len = spec["anthropometry"]["segments"]["head"]["length_m"]
    skull = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "visual_head_skull")
    centre = data.geom_xpos[skull]  # skull centre sits 0.58 L above the neck point
    assert centre[2] - neck[2] == pytest.approx(0.58 * head_len, abs=0.005)
    assert np.hypot(*(centre - neck)[:2]) < 0.005  # centred over the neck
    sole = min(
        data.geom_xpos[i][2] - model.geom_size[i][0]
        for i in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith(
            "contact_"
        )
    )
    stature = neck[2] + head_len - sole
    assert stature == pytest.approx(
        spec["anthropometry"].get("stature_m", 1.71), rel=0.06
    )


def test_face_points_forward_with_the_body() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(_export(DRIVER, None)[0])
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    nose = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "visual_head_nose")
    skull = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "visual_head_skull")
    toes = [model.body(n).id for n in ("toes_l", "toes_r")]
    heels = [model.body(n).id for n in ("calcn_l", "calcn_r")]
    foot_dir = data.xpos[toes].mean(axis=0) - data.xpos[heels].mean(axis=0)
    assert (data.geom_xpos[nose] - data.geom_xpos[skull])[:2] @ foot_dir[:2] > 0


def test_spec_without_head_body_gets_a_torso_head() -> None:
    pytest.importorskip("mujoco")
    _, meta = _export(BODIES / "full_body_spec_v1.json", None)
    head = meta["visual_layer"]["head"]
    assert head["enabled"] and head["source"] == "torso"


def test_orientation_override_uses_a_massless_mocap_body() -> None:
    mujoco = pytest.importorskip("mujoco")
    plain = mujoco.MjModel.from_xml_string(
        exporter.export_full_body_mjcf(DRIVER.read_bytes())[0]
    )
    xml, meta = _export(
        DRIVER, {"orientation_override": {"frame": "world", "yaw_rad": 0.4}}
    )
    model = mujoco.MjModel.from_xml_string(xml)
    assert meta["visual_layer"]["head"]["mode"] == "mocap"
    assert (model.nq, model.nv) == (plain.nq, plain.nv)
    np.testing.assert_array_equal(model.body_mass[: plain.nbody], plain.body_mass)
    np.testing.assert_array_equal(model.body_inertia[: plain.nbody], plain.body_inertia)
    data = _state(mujoco, model)
    drive_visual_head(model, data, 0.5, 0.0, 0.0, "parent")
    mujoco.mj_forward(model, data)
    mocap = model.body("visual_head")
    site = model.site("visual_head_anchor").id
    np.testing.assert_allclose(data.xpos[mocap.id], data.site_xpos[site], atol=1e-9)
    # yaw 0.5 rad left of the physics head's frame
    expected = data.site_xmat[site].reshape(3, 3) @ np.array(
        [[np.cos(0.5), -np.sin(0.5), 0], [np.sin(0.5), np.cos(0.5), 0], [0, 0, 1]]
    )
    np.testing.assert_allclose(data.xmat[mocap.id].reshape(3, 3), expected, atol=1e-9)
    with pytest.raises(ValueError):
        drive_visual_head(model, data, frame="sideways")


def test_meshes_body_model_is_rejected_until_available() -> None:
    pytest.importorskip("mujoco")
    doc = document_from_dict(
        {"schema_version": "appearance-v1", "body_model": "meshes"}
    )
    with pytest.raises(ValueError, match="rigid-bind"):
        exporter.export_full_body_mjcf(DRIVER.read_bytes(), appearance=doc)


def test_blend_balls_are_no_larger_than_the_adjoining_limb() -> None:
    from src.engines.physics_engines.mujoco.python import appearance_layer
    from src.shared.python.model_appearance import library
    from src.shared.python.motion_matching.visual_skeleton import derive_visual_skeleton

    assert library.blend_scale(True) <= 1.1 and library.blend_scale(False) <= 1.1
    spec = json.loads(DRIVER.read_text())
    doc = document_from_dict({"schema_version": "appearance-v1"})
    caps: dict[str, list] = {}
    for cap in derive_visual_skeleton(spec).capsules:
        caps.setdefault(cap.body, []).append(cap)
    seen = 0
    for body, group in caps.items():
        part = library.classify_body(body)
        if part not in library.BLEND_PARTS:
            continue
        for label, mesh, _ in appearance_layer._body_meshes(doc, body, group):
            if label.startswith("blend"):
                seen += 1
                extent = np.ptp(mesh.vertices, axis=0).max() / 2.0
                assert extent <= 1.1 * library.PART_RADIUS_M[part] + 1e-9
    assert seen >= 4  # both shoulders and both hips
