"""Unit tests for motion_pipeline.model_bridge (epic #8390, B2/#8397)."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from src.shared.python.motion_pipeline.contracts import (
    JointDef,
    JointLimit,
    SkeletonRig,
)
from src.shared.python.motion_pipeline.model_bridge import (
    rig_root_link_name,
    rig_to_urdf,
)

pytestmark = pytest.mark.unit


def _rig() -> SkeletonRig:
    joints = {
        "hip": JointDef(
            name="hip",
            parent=None,
            children=["knee"],
            tpose_offset=[0.0, 0.0, 1.0],
            axes=["X", "Y"],
            limits=[
                JointLimit(lower=-1.0, upper=1.0),
                JointLimit(lower=-0.5, upper=0.5),
            ],
        ),
        "knee": JointDef(
            name="knee",
            parent="hip",
            children=[],
            tpose_offset=[0.0, 0.0, -0.4],
            axes=["X"],
        ),
    }
    return SkeletonRig(id="test_rig", joints=joints, root_joint="hip")


def test_urdf_parses_and_counts_one_joint_per_dof() -> None:
    rig = _rig()
    root = ET.fromstring(rig_to_urdf(rig))
    assert root.tag == "robot"
    revolute = [j for j in root.findall("joint") if j.get("type") == "revolute"]
    assert len(revolute) == rig.num_dofs == 3


def test_urdf_joint_order_matches_rig_dof_order() -> None:
    root = ET.fromstring(rig_to_urdf(_rig()))
    names = [j.get("name") for j in root.findall("joint")]
    assert names == ["hip_dof0", "hip_dof1", "knee_dof0"]


def test_urdf_applies_limits_and_default_limits() -> None:
    root = ET.fromstring(rig_to_urdf(_rig()))
    limits = {j.get("name"): j.find("limit") for j in root.findall("joint")}
    assert float(limits["hip_dof0"].get("lower")) == -1.0
    assert float(limits["hip_dof1"].get("upper")) == 0.5
    # knee has no explicit limit -> defaults to +/- pi
    assert float(limits["knee_dof0"].get("lower")) == pytest.approx(-3.14159, abs=1e-3)


def test_urdf_offset_applied_on_first_dof_only() -> None:
    root = ET.fromstring(rig_to_urdf(_rig()))
    origins = {
        j.get("name"): j.find("origin").get("xyz") for j in root.findall("joint")
    }
    assert origins["hip_dof0"] == "0.0 0.0 1.0"
    assert origins["hip_dof1"] == "0.0 0.0 0.0"


def test_urdf_has_transmission_per_dof() -> None:
    root = ET.fromstring(rig_to_urdf(_rig()))
    assert len(root.findall("transmission")) == 3


def test_root_link_name_matches_helper() -> None:
    rig = _rig()
    root = ET.fromstring(rig_to_urdf(rig))
    link_names = {ln.get("name") for ln in root.findall("link")}
    assert rig_root_link_name(rig) in link_names


def test_zero_dof_rig_rejected() -> None:
    # SkeletonRig contracts require joints, so simulate emptiness via a
    # minimal object exposing the accessed attributes.
    class _FakeRig:
        num_dofs = 0
        id = "empty"
        joints: dict = {}

    with pytest.raises(ValueError, match="at least one DOF"):
        rig_to_urdf(_FakeRig())  # type: ignore[arg-type]


def test_link_inertials_emit_origin_mass_and_full_tensor() -> None:
    from src.shared.python.motion_pipeline.model_bridge import LinkInertial

    rig = _rig()
    knee = LinkInertial(
        mass=2.5, com=(0.0, 0.0, -0.2), inertia=(0.03, 0.03, 0.001, 0.0, 0.0, 0.0)
    )
    root = ET.fromstring(rig_to_urdf(rig, link_inertials={"knee": knee}))
    links = {ln.get("name"): ln for ln in root.findall("link")}
    inertial = links["knee_link"].find("inertial")
    assert inertial.find("origin").get("xyz") == "0.0 0.0 -0.2"
    assert float(inertial.find("mass").get("value")) == 2.5
    assert float(inertial.find("inertia").get("izz")) == pytest.approx(0.001)
    # Joints without an entry keep the generic conditioning placeholder.
    hip_mass = links["hip_link"].find("inertial").find("mass").get("value")
    assert float(hip_mass) == 1.0
    # Intermediate multi-axis carrier links are never overridden.
    dof0 = links["hip_dof0_link"].find("inertial").find("mass").get("value")
    assert float(dof0) == pytest.approx(1e-2)


def test_link_inertials_default_output_is_unchanged() -> None:
    rig = _rig()
    assert rig_to_urdf(rig) == rig_to_urdf(rig, link_inertials={})


def test_link_inertials_unknown_joint_rejected() -> None:
    from src.shared.python.motion_pipeline.model_bridge import LinkInertial

    with pytest.raises(ValueError, match="not in rig"):
        rig_to_urdf(_rig(), link_inertials={"ankle": LinkInertial(mass=1.0)})


def test_link_inertial_contracts() -> None:
    from src.shared.python.motion_pipeline.model_bridge import LinkInertial

    with pytest.raises(ValueError, match="mass"):
        LinkInertial(mass=0.0)
    with pytest.raises(ValueError, match="non-negative"):
        LinkInertial(mass=1.0, inertia=(-1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    tensor = LinkInertial(mass=1.0, inertia=(1, 2, 3, 4, 5, 6)).inertia_matrix
    assert tensor == [[1, 4, 5], [4, 2, 6], [5, 6, 3]]
