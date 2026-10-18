"""full-body-v2 schema: regional lumbar and thoracic trunk joints (HRG-D1, #13170)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.model_appearance.skinned_body import spec_body_poses
from src.shared.python.motion_matching import full_body_v2 as v2
from src.shared.python.motion_matching.full_body_spec import (
    canonical_sha256,
    load_full_body_spec,
    order_full_body_joints,
    save_full_body_spec,
    upper_body_slice,
    validate_full_body_spec,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
V1_PATH = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
PELVIS = "solid_reference:GolfSwing3D_Kinetic/Hips and Torso Inputs/LowerTorso"
THORAX = "solid_reference:GolfSwing3D_Kinetic/Hips and Torso Inputs/COMRod"
DUMMY = "solid_reference:GolfSwing3D_Kinetic/Hips and Torso Inputs/UpperTorsoBase"
V1_TRUNK = ("SpineInputX", "SpineInputY", "TorsoInput")


def _v1() -> dict:
    return json.loads(V1_PATH.read_text(encoding="utf-8"))


def _v2(v1: dict | None = None) -> dict:
    v1 = _v1() if v1 is None else v1
    return v2.derive_full_body_v2_spec(
        v1, v2.segment_boundary_locations(v1), provenance="HRG-D1 unit fixture"
    )


def _mass(spec: dict) -> float:
    return sum(s["mass_kg"] for b in spec["bodies"] for s in b["solids"])


def test_derivation_never_mutates_the_v1_document() -> None:
    v1 = _v1()
    before = copy.deepcopy(v1)
    sha = canonical_sha256(v1)
    doc = _v2(v1)
    assert v1 == before
    assert canonical_sha256(v1) == sha
    assert doc["source_spec_sha256"] == sha
    assert doc["base_upper_body_sha256"] == v1["upper_body_sha256"]
    assert canonical_sha256(doc) != sha


def test_v2_document_validates_and_round_trips(tmp_path: Path) -> None:
    doc = _v2()
    assert doc["schema_version"] == "full-body-v2"
    upper = upper_body_slice(doc)
    assert upper["schema_version"] == v2.V2_UPPER_BODY_SCHEMA_VERSION
    assert doc["upper_body_sha256"] == canonical_sha256(upper)
    validate_full_body_spec(doc, upper)
    path = save_full_body_spec(doc, tmp_path / "full_body_v2.json")
    loaded = load_full_body_spec(path, upper)
    assert canonical_sha256(loaded) == canonical_sha256(doc)


def test_v2_coordinate_order_replaces_the_v1_trunk_in_place() -> None:
    v1, doc = _v1(), _v2()
    order = doc["coordinate_order"]
    first = v1["coordinate_order"].index(V1_TRUNK[0])
    assert order[:first] == v1["coordinate_order"][:first]
    assert tuple(order[first : first + 6]) == v2.V2_TRUNK.coordinates
    rest = [c for c in v1["coordinate_order"] if c not in V1_TRUNK]
    assert [c for c in order if c not in v2.V2_TRUNK.coordinates] == rest
    assert v2.v2_coordinate_order(v1["coordinate_order"]) == order
    assert doc["upper_body_counts"]["coordinates"] == (
        v1["upper_body_counts"]["coordinates"] + 3
    )
    assert not set(V1_TRUNK) & set(order)


def test_v2_topology_is_pelvis_lumbar_thorax_and_drops_the_dummy() -> None:
    doc = _v2()
    joints = {j["child"]: j for j in doc["joints"]}
    lumbar = joints[v2.LUMBAR_BODY]
    thoracic = joints[THORAX]
    assert lumbar["parent"] == PELVIS
    assert thoracic["parent"] == v2.LUMBAR_BODY
    assert [p["primitive"] for p in lumbar["primitives"]] == ["Ry", "Rx", "Rz"]
    assert [p["coordinate"] for p in thoracic["primitives"]] == list(
        v2.V2_TRUNK.coordinates[3:]
    )
    bodies = {b["name"]: b for b in doc["bodies"]}
    assert DUMMY not in bodies
    lumbar_solids = [
        s["name"].rsplit("/", 1)[-1] for s in bodies[v2.LUMBAR_BODY]["solids"]
    ]
    thorax_solids = [s["name"].rsplit("/", 1)[-1] for s in bodies[THORAX]["solids"]]
    assert lumbar_solids == ["middle_trunk"]
    assert thorax_solids == ["upper_trunk"]
    assert _mass(doc) == pytest.approx(_mass(_v1()) - 0.3, abs=1e-12)
    assert doc["trunk_topology"]["dropped_bodies"] == [{"body": DUMMY, "mass_kg": 0.3}]
    ordered = [j["name"] for j in order_full_body_joints(doc)]
    assert ordered.index(lumbar["name"]) < ordered.index(thoracic["name"])


def test_neutral_pose_of_every_retained_body_and_solid_is_unchanged() -> None:
    v1, doc = _v1(), _v2()
    p1, p2 = spec_body_poses(v1), spec_body_poses(doc)
    for name in p2:
        if name in p1:
            np.testing.assert_allclose(p2[name], p1[name], atol=1e-12)

    def solid_world(spec: dict, poses: dict) -> dict:
        return {
            s["name"]: poses[b["name"]] @ np.asarray(s["placement"])
            for b in spec["bodies"]
            for s in b["solids"]
        }

    w1, w2 = solid_world(v1, p1), solid_world(doc, p2)
    for name, pose in w2.items():
        np.testing.assert_allclose(pose, w1[name], atol=1e-12)
    frames1 = {
        f["name"]: p1[f["body"]] @ np.asarray(f["placement"]) for f in v1["frames"]
    }
    for frame in doc["frames"]:
        np.testing.assert_allclose(
            p2[frame["body"]] @ np.asarray(frame["placement"]),
            frames1[frame["name"]],
            atol=1e-12,
        )


def test_isb_mapping_and_soft_limits_are_declared_in_isb_spine_sign() -> None:
    trunk = v2.V2_TRUNK
    assert trunk.isb_axis("lumbar_flexion") == ("flexion_extension", -1.0)
    assert trunk.isb_axis("thoracic_lateral_bending") == ("lateral_bending", 1.0)
    assert trunk.isb_axis("thoracic_axial_rotation") == ("axial_rotation", 1.0)
    lumbar, thoracic = trunk.joints
    assert lumbar.soft_limits_isb_deg == ((-40.0, 15.0), (-20.0, 20.0), (-15.0, 15.0))
    assert thoracic.soft_limits_isb_deg == ((-30.0, 15.0), (-25.0, 25.0), (-50.0, 50.0))
    ranges = v2.v2_trunk_ranges_deg()
    # Model sign: positive lumbar_flexion is flexion, so ISB FE -40..+15 is -15..+40.
    assert ranges["lumbar_flexion"] == (-15.0, 40.0)
    assert ranges["thoracic_axial_rotation"] == (-50.0, 50.0)
    doc = _v2()
    for name, bounds in ranges.items():
        assert tuple(doc["coordinate_ranges_deg"][name]) == bounds
    assert not set(V1_TRUNK) & set(doc["coordinate_ranges_deg"])


def test_positive_coordinates_move_the_thorax_as_labelled() -> None:
    """Probe: +10 deg on each v2 coordinate moves a thorax point as its label says."""
    doc = _v2()
    rest = spec_body_poses(doc)
    probe = np.array([0.0, 0.0, 0.3, 1.0])  # 0.3 m up the thorax
    expected = {
        "flexion": (0, 1.0),  # top moves forward (+x)
        "lateral_bending": (1, -1.0),  # top moves right (-y)
        "axial_rotation": (None, None),
    }
    for name in v2.V2_TRUNK.coordinates:
        moved = spec_body_poses(doc, {name: np.radians(10.0)})
        delta = (moved[THORAX] @ probe - rest[THORAX] @ probe)[:3]
        kind = name.split("_", 1)[1]
        axis, sign = expected[kind]
        if axis is None:
            # Left axial rotation: a forward probe turns toward +y.
            fwd = np.array([0.2, 0.0, 0.3, 1.0])
            turn = (moved[THORAX] @ fwd - rest[THORAX] @ fwd)[:3]
            assert turn[1] > 0.0, name
        else:
            assert sign * delta[axis] > 0.0, name


def test_v2_validator_rejects_a_broken_trunk() -> None:
    doc = _v2()
    upper = upper_body_slice(doc)
    broken = copy.deepcopy(doc)
    lumbar = next(j for j in broken["joints"] if j["child"] == v2.LUMBAR_BODY)
    lumbar["primitives"][0]["primitive"] = "Rx"
    with pytest.raises(ValueError, match="trunk"):
        v2.validate_full_body_v2_spec(broken, upper_body_slice(broken))
    with pytest.raises(ValueError, match="identity"):
        validate_full_body_spec(doc, upper_body_slice(_v1()))
    tampered = copy.deepcopy(doc)
    tampered["source_spec_sha256"] = "0" * 63
    with pytest.raises(ValueError, match="source_spec_sha256"):
        validate_full_body_spec(tampered, upper)


def test_derivation_preconditions() -> None:
    v1 = _v1()
    with pytest.raises(ValueError, match="full-body-v1"):
        v2.derive_full_body_v2_spec(
            _v2(), v2.segment_boundary_locations(v1), provenance="x"
        )
    with pytest.raises(ValueError, match="provenance"):
        v2.derive_full_body_v2_spec(
            v1, v2.segment_boundary_locations(v1), provenance=" "
        )
    native = json.loads(
        (ROOT / "docs/development/full_body_models/full_body_spec_v1.json").read_text()
    )
    with pytest.raises(ValueError, match="middle_trunk"):
        v2.segment_boundary_locations(native)
    with pytest.raises(ValueError, match="finite"):
        v2.TrunkJointLocations((0.0, 0.0, float("nan")), (0.0, 0.0, 0.3), "x")
    with pytest.raises(ValueError, match="distinct"):
        v2.TrunkJointLocations((0.0, 0.0, 0.1), (0.0, 0.0, 0.1), "x")


def test_segment_boundary_locations_are_the_pivot_and_the_xiphion() -> None:
    v1 = _v1()
    loc = v2.segment_boundary_locations(v1)
    twist = next(
        j for j in v1["joints"] if j["primitives"][0]["coordinate"] == "TorsoInput"
    )
    pivot = np.asarray(twist["parent_to_base"])[:3, 3]
    np.testing.assert_allclose(loc.l5s1_in_pelvis_m, pivot, atol=1e-15)
    hub = next(f for f in v1["frames"] if f["name"] == "Hub")
    hub_z = np.asarray(hub["placement"])[2, 3]
    np.testing.assert_allclose(
        loc.t12l1_in_pelvis_m, pivot + np.array([0.0, 0.0, 0.5 * hub_z]), atol=1e-15
    )
    assert "HRG-D2" in loc.provenance
