"""Full-body specification: upper-body slice identity, validation, hashing (FB-1)."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching import full_body_spec as module

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
UPPER = (
    ROOT
    / "docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json"
)


def _upper() -> dict:
    return json.loads(UPPER.read_text())


def _eye() -> list[list[float]]:
    return np.eye(4).tolist()


def _extension() -> module.LowerLimbExtension:
    pelvis = "solid_reference:GolfSwing3D_Kinetic/Hips and Torso Inputs/LowerTorso"
    bodies = [
        module.BodySpec(
            "femur_r", 9.8, (0.0, -0.17, 0.0), (0.139, 0.036, 0.147, 0.0, 0.0, 0.0)
        ),
        module.BodySpec(
            "tibia_r", 3.9, (0.0, -0.20, 0.0), (0.063, 0.006, 0.063, 0.0, 0.0, 0.0)
        ),
    ]
    joints = [
        module.JointSpec(
            "hip_r",
            pelvis,
            "femur_r",
            _eye(),
            _eye(),
            ("Rx", "Ry", "Rz"),
            ("hip_flexion_r", "hip_adduction_r", "hip_rotation_r"),
        ),
        module.JointSpec(
            "knee_r", "femur_r", "tibia_r", _eye(), _eye(), ("Rz",), ("knee_angle_r",)
        ),
    ]
    return module.LowerLimbExtension(bodies, joints, provenance="unit test")


def _contact() -> module.ContactSpec:
    return module.ContactSpec(
        law="hunt_crossley_coulomb",
        parameters={
            "stiffness_n_m": 5e4,
            "dissipation_s_m": 1.0,
            "static_friction": 0.9,
            "dynamic_friction": 0.8,
            "viscous_friction": 0.0,
            "transition_velocity_m_s": 0.05,
        },
        spheres=(module.ContactSphere("heel_r", "tibia_r", (0.0, -0.45, 0.0), 0.03),),
        ground_normal_policy="opposite_gravity",
        ground_height_m=None,
        provenance="unit test",
    )


def _markers() -> dict[str, module.MarkerAttachment]:
    return {"RKneeOut": module.MarkerAttachment("femur_r", None)}


def test_derived_document_preserves_upper_body_slice_and_appends() -> None:
    upper = _upper()
    doc = module.derive_full_body_spec(
        upper, _extension(), _contact(), _markers(), provenance="test"
    )
    assert doc["schema_version"] == module.FULL_BODY_SCHEMA_VERSION
    assert doc["coordinate_order"][:27] == upper["coordinate_order"]
    assert doc["coordinate_order"][27:] == [
        "hip_flexion_r",
        "hip_adduction_r",
        "hip_rotation_r",
        "knee_angle_r",
    ]
    assert doc["upper_body_sha256"] == module.canonical_sha256(upper)
    assert module.upper_body_slice(doc) == upper
    assert len(doc["bodies"]) == len(upper["bodies"]) + 2
    assert len(doc["joints"]) == len(upper["joints"]) + 2
    module.validate_full_body_spec(doc, upper)


def test_hash_is_order_independent_and_roundtrips(tmp_path: Path) -> None:
    doc = module.derive_full_body_spec(
        _upper(), _extension(), _contact(), _markers(), provenance="test"
    )
    shuffled = json.loads(json.dumps(doc, sort_keys=True))
    assert module.canonical_sha256(doc) == module.canonical_sha256(shuffled)
    path = tmp_path / "spec.json"
    module.save_full_body_spec(doc, path)
    again = module.load_full_body_spec(path, _upper())
    assert module.canonical_sha256(again) == module.canonical_sha256(doc)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["coordinate_order"].__setitem__(0, "renamed"),
        lambda d: d["coordinate_order"].append("hip_flexion_r"),
        lambda d: d["bodies"][-1]["solids"][0].__setitem__("mass_kg", -1.0),
        lambda d: d["bodies"][-1]["solids"][0].__setitem__(
            "inertia_com_kg_m2", [[1, 0, 0], [0, -1, 0], [0, 0, 1]]
        ),
        lambda d: d["contact"].__setitem__("provenance", ""),
        lambda d: d["marker_attachments"].__setitem__(
            "NotALabel", {"body": "femur_r", "offset_m": None}
        ),
        lambda d: d["marker_attachments"]["RKneeOut"].__setitem__(
            "body", "no_such_body"
        ),
        lambda d: d["joints"][-1].__setitem__("parent", "no_such_body"),
        lambda d: d["gravity_m_s2"].__setitem__(2, 0.0),
    ],
)
def test_validator_rejects_corruptions(mutate) -> None:
    upper = _upper()
    doc = module.derive_full_body_spec(
        upper, _extension(), _contact(), _markers(), provenance="test"
    )
    bad = copy.deepcopy(doc)
    mutate(bad)
    with pytest.raises(ValueError):
        module.validate_full_body_spec(bad, upper)


def test_body_and_joint_specs_validate_inputs() -> None:
    with pytest.raises(ValueError):
        module.BodySpec("x", 0.0, (0, 0, 0), (1, 1, 1, 0, 0, 0))
    with pytest.raises(ValueError):
        module.BodySpec("x", 1.0, (0, 0, 0), (1, 1, -1, 0, 0, 0))
    with pytest.raises(ValueError):
        module.JointSpec("j", "a", "b", _eye(), _eye(), ("Qx",), ("c",))
    with pytest.raises(ValueError):
        module.JointSpec("j", "a", "b", _eye(), _eye(), ("Rx", "Ry"), ("c",))
    with pytest.raises(ValueError):
        module.ContactSphere("s", "b", (0, 0, 0), 0.0)


def test_pelvis_alignment_recovers_a_known_transform() -> None:
    rng = np.random.default_rng(3)
    local = rng.normal(size=(4, 3))
    angle = 0.4
    rot = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    trans = np.array([0.1, -0.2, 0.3])
    other = local @ rot.T + trans
    transform, residual = module.pelvis_alignment(
        dict(zip("abcd", other, strict=True)), dict(zip("abcd", local, strict=True))
    )
    np.testing.assert_allclose(transform[:3, :3], rot, atol=1e-12)
    np.testing.assert_allclose(transform[:3, 3], trans, atol=1e-12)
    assert residual < 1e-12
    with pytest.raises(ValueError):
        module.pelvis_alignment({"a": other[0]}, {"a": local[0]})


def test_committed_full_body_spec_validates_against_the_qualified_base() -> None:
    path = ROOT / "docs/development/full_body_models/full_body_spec_v1.json"
    receipt = json.loads(
        (ROOT / "docs/development/full_body_models/build_receipt.json").read_text()
    )
    doc = module.load_full_body_spec(path, _upper())
    assert module.canonical_sha256(doc) == receipt["spec_sha256"]
    assert len(doc["coordinate_order"]) == 41
    assert doc["coordinate_order"][:27] == _upper()["coordinate_order"]
    assert {b["name"] for b in doc["bodies"]} >= {"femur_l", "calcn_r", "toes_l"}
    assert doc["contact"]["ground"]["calibrated"] is False
    assert receipt["pelvis_alignment"]["rms_residual_m"] < 0.01


def test_order_full_body_joints_orders_upper_tree_then_lower_chains() -> None:
    path = ROOT / "docs/development/full_body_models/full_body_spec_v1.json"
    doc = module.load_full_body_spec(path, _upper())
    ordered = module.order_full_body_joints(doc)
    assert len(ordered) == len(doc["joints"])
    upper_joints = {j["name"] for j in module.upper_body_slice(doc)["joints"]}
    # Upper joints come first
    for j in ordered[: len(upper_joints)]:
        assert j["name"] in upper_joints
    # Lower limb joints follow
    lower_names = [j["name"] for j in ordered[len(upper_joints) :]]
    assert lower_names[0] == "hip_r"
    assert "knee_r" in lower_names
    assert "hip_l" in lower_names
