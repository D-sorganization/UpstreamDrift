"""Real MuJoCo kinematics only: spherical export is not dynamics qualification."""

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.native_mjcf import export_native_mjcf
from src.engines.physics_engines.mujoco.python.native_spherical_mjcf import (
    export_native_spherical_mjcf,
)
from src.shared.python.pose_interchange.native_joint_state import (
    NativeJointStateAdapter,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def raw() -> bytes:
    return (
        Path(__file__).resolve().parents[3]
        / "docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json"
    ).read_bytes()


def test_export_preserves_every_nonjoint_element_and_identity(raw: bytes) -> None:
    original, metadata = export_native_mjcf(raw)
    xml, spherical = export_native_spherical_mjcf(raw)
    assert (
        spherical["model_sha256"]
        == metadata["model_sha256"]
        == hashlib.sha256(raw).hexdigest()
    )
    assert spherical["representation"] == "native-spherical-mjcf-v1"
    assert (
        spherical["representation_sha256"] == hashlib.sha256(xml.encode()).hexdigest()
    )
    assert spherical["representation_sha256"] != metadata["mjcf_sha256"]
    assert spherical["coordinate_order"] == metadata["coordinate_order"]
    assert spherical["frame_sites"] == metadata["frame_sites"]
    assert spherical["expected_nq"] == 30 and spherical["expected_nv"] == 27
    assert "unqualified" in spherical["execution"]
    trees = [ET.fromstring(text) for text in (original, xml)]
    adapter = NativeJointStateAdapter(json.loads(raw))
    assert spherical["specification_sha256"] == adapter.specification_sha256
    assert set(spherical["ball_joints"]) == {group.name for group in adapter.groups}
    for group in adapter.groups:
        entry = spherical["ball_joints"][group.name]
        assert tuple(entry["native_coordinates"]) == group.coordinates
        assert entry["axes"] == group.axes
        assert entry["body"] == group.child_body
    for name in adapter.scalar_coordinates:
        matches = [
            next(j for j in tree.iter("joint") if j.get("name") == name)
            for tree in trees
        ]
        assert matches[0].attrib == matches[1].attrib
    for tree in trees:
        for body in tree.iter("body"):
            for joint in list(body.findall("joint")):
                body.remove(joint)
    assert ET.tostring(trees[0]) == ET.tostring(trees[1])
    assert export_native_mjcf(raw) == (original, metadata)


@pytest.mark.parametrize("damage", ["duplicate", "missing_group", "axes"])
def test_invalid_native_group_inventory_rejected(raw: bytes, damage: str) -> None:
    spec = json.loads(raw)
    if damage == "duplicate":
        spec["coordinate_order"][1] = spec["coordinate_order"][0]
    else:
        group = next(
            j
            for j in spec["joints"]
            if any(p["coordinate"] == "LSInputX" for p in j["primitives"])
        )
        if damage == "missing_group":
            group["primitives"].pop()
            spec["coordinate_order"].remove("LSInputZ")
        else:
            group["primitives"][0]["primitive"] = "Rz"
            group["primitives"][2]["primitive"] = "Rx"
    with pytest.raises(ValueError):
        export_native_spherical_mjcf(json.dumps(spec).encode())


@pytest.mark.live_simulation
def test_real_mujoco_compile_and_multisample_body_site_parity(raw: bytes) -> None:
    mj = pytest.importorskip("mujoco")
    original, _ = export_native_mjcf(raw)
    xml, metadata = export_native_spherical_mjcf(raw)
    models = [mj.MjModel.from_xml_string(text) for text in (original, xml)]
    native, alternate = models
    assert (native.nq, native.nv, alternate.nq, alternate.nv) == (27, 27, 30, 27)
    assert alternate.njnt == 21
    assert np.count_nonzero(alternate.jnt_type == mj.mjtJoint.mjJNT_BALL) == 3
    for attribute in (
        "body_mass",
        "body_inertia",
        "body_ipos",
        "body_iquat",
        "site_pos",
        "site_quat",
        "eq_data",
        "eq_type",
        "eq_obj1id",
        "eq_obj2id",
        "eq_solimp",
        "eq_solref",
    ):
        np.testing.assert_array_equal(
            getattr(native, attribute), getattr(alternate, attribute)
        )
    np.testing.assert_array_equal(native.opt.gravity, alternate.opt.gravity)
    for attribute in (
        "dof_damping",
        "dof_armature",
        "dof_frictionloss",
        "jnt_limited",
        "jnt_stiffness",
    ):
        assert np.count_nonzero(getattr(alternate, attribute)) == 0
    adapter = NativeJointStateAdapter(json.loads(raw))
    assert tuple(metadata["scalar_coordinates"]) == adapter.scalar_coordinates
    assert len(adapter.scalar_coordinates) == 18
    data = [mj.MjData(model) for model in models]
    for index in range(3):
        q = dict.fromkeys(adapter.coordinate_order, 0.1 * index)
        for group in adapter.groups:
            for name, value in zip(
                group.coordinates, [6.4 + index / 10, -2 - index / 10, 0.4], strict=True
            ):
                q[name] = value
        zeros = dict.fromkeys(q, 0.0)
        state = adapter.export(q, zeros, zeros, zeros)
        for name in adapter.coordinate_order:
            data[0].qpos[native.jnt_qposadr[native.joint(name).id]] = q[name]
        for name in adapter.scalar_coordinates:
            data[1].qpos[alternate.jnt_qposadr[alternate.joint(name).id]] = (
                state.scalars[name][0]
            )
        for group_name, entry in metadata["ball_joints"].items():
            start = alternate.jnt_qposadr[alternate.joint(entry["joint_name"]).id]
            data[1].qpos[start : start + 4] = state.rotations[
                group_name
            ].quaternion_wxyz
        for model, item in zip(models, data, strict=True):
            mj.mj_kinematics(model, item)
        for attribute in ("xpos", "xmat", "site_xpos", "site_xmat"):
            np.testing.assert_allclose(
                getattr(data[0], attribute),
                getattr(data[1], attribute),
                atol=2e-14,
                rtol=0,
            )
