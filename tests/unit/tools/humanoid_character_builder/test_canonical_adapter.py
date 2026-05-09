"""Test the HCB→canonical adapter (issue #4601 foundation).

Verifies that the adapter in
``humanoid_character_builder.generators._canonical_adapter`` correctly
converts HCB domain types (``GeneratedLink``, ``GeneratedJoint``) to
canonical model_generation types (``Link``, ``Joint``) and that the
resulting URDF emitted via the canonical writer is well-formed.

This is the structural foundation for #4601 — full byte-identical
migration is tracked as a follow-up.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest
from humanoid_character_builder import BodyParameters, CharacterBuilder
from humanoid_character_builder.generators._canonical_adapter import (
    to_canonical_lists,
    write_humanoid_urdf_via_canonical,
)


@pytest.fixture(scope="module")
def humanoid_model_data() -> tuple[dict, list]:
    """Build a humanoid model and return its (links, joints) collections."""
    builder = CharacterBuilder()
    params = BodyParameters(height_m=1.80, mass_kg=80.0)
    # Use the internal generator path to get GeneratedLink/GeneratedJoint
    # collections without going through XML emission first.
    from humanoid_character_builder.generators.urdf_generator import (
        HumanoidURDFGenerator,
        URDFGeneratorConfig,
    )

    gen = HumanoidURDFGenerator(URDFGeneratorConfig())
    # Many generators expose a `_build_model` or call into core/model.py.
    # We round-trip via the public XML path to extract links/joints from
    # the model object the generator builds internally.
    gen.generate(params)  # populate any internal state
    # Retrieve the materials/links/joints directly from the builder; the
    # adapter only needs the canonical forms, so reach into the generator.
    return getattr(gen, "_last_links", {}), getattr(gen, "_last_joints", [])


@pytest.mark.unit
def test_adapter_smoke_with_synthetic_inputs() -> None:
    """Build a tiny synthetic GeneratedLink/GeneratedJoint and convert."""
    from humanoid_character_builder.core.model import (
        GeneratedJoint,
        GeneratedLink,
    )
    from humanoid_character_builder.mesh.inertia_calculator import (
        InertiaResult,
    )

    inertia = InertiaResult(ixx=0.01, iyy=0.01, izz=0.01, mass=1.0, volume=0.001)
    link = GeneratedLink(
        name="base",
        mass=1.0,
        inertia=inertia,
        visual_geometry={"type": "box", "size": (0.1, 0.1, 0.1)},
        collision_geometry={"type": "box", "size": (0.1, 0.1, 0.1)},
        origin_xyz=(0.0, 0.0, 0.0),
        origin_rpy=(0.0, 0.0, 0.0),
    )
    link2 = GeneratedLink(
        name="arm",
        mass=0.5,
        inertia=inertia,
        visual_geometry={"type": "cylinder", "radius": 0.02, "length": 0.2},
        collision_geometry={"type": "cylinder", "radius": 0.02, "length": 0.2},
        origin_xyz=(0.0, 0.0, 0.0),
        origin_rpy=(0.0, 0.0, 0.0),
    )
    joint = GeneratedJoint(
        name="base_to_arm",
        joint_type="revolute",
        parent="base",
        child="arm",
        origin_xyz=(0.1, 0.0, 0.0),
        origin_rpy=(0.0, 0.0, 0.0),
        axis=(1.0, 0.0, 0.0),
        limits={"lower": -1.0, "upper": 1.0, "effort": 10.0, "velocity": 5.0},
        dynamics={"damping": 0.1, "friction": 0.05},
    )

    canon_links, canon_joints = to_canonical_lists(
        {"base": link, "arm": link2}, [joint]
    )

    assert len(canon_links) == 2
    assert {link.name for link in canon_links} == {"base", "arm"}
    assert len(canon_joints) == 1
    assert canon_joints[0].name == "base_to_arm"
    assert canon_joints[0].limits is not None
    assert canon_joints[0].limits.lower == -1.0
    assert canon_joints[0].limits.upper == 1.0


@pytest.mark.unit
def test_canonical_writer_emits_valid_urdf_from_synthetic_input() -> None:
    """End-to-end: HCB types → canonical types → canonical URDFWriter → valid XML."""
    from humanoid_character_builder.core.model import (
        GeneratedJoint,
        GeneratedLink,
    )
    from humanoid_character_builder.mesh.inertia_calculator import (
        InertiaResult,
    )

    inertia = InertiaResult(ixx=0.01, iyy=0.01, izz=0.01, mass=1.0, volume=0.001)
    link = GeneratedLink(
        name="base",
        mass=1.0,
        inertia=inertia,
        visual_geometry={"type": "box", "size": (0.1, 0.1, 0.1)},
        collision_geometry={"type": "box", "size": (0.1, 0.1, 0.1)},
        origin_xyz=(0.0, 0.0, 0.0),
        origin_rpy=(0.0, 0.0, 0.0),
    )
    link2 = GeneratedLink(
        name="arm",
        mass=0.5,
        inertia=inertia,
        visual_geometry={"type": "cylinder", "radius": 0.02, "length": 0.2},
        collision_geometry={"type": "cylinder", "radius": 0.02, "length": 0.2},
        origin_xyz=(0.0, 0.0, 0.0),
        origin_rpy=(0.0, 0.0, 0.0),
    )
    joint = GeneratedJoint(
        name="base_to_arm",
        joint_type="revolute",
        parent="base",
        child="arm",
        origin_xyz=(0.1, 0.0, 0.0),
        origin_rpy=(0.0, 0.0, 0.0),
        axis=(1.0, 0.0, 0.0),
        limits={"lower": -1.0, "upper": 1.0, "effort": 10.0, "velocity": 5.0},
        dynamics={"damping": 0.1, "friction": 0.05},
    )

    urdf_xml = write_humanoid_urdf_via_canonical(
        robot_name="test_humanoid",
        links={"base": link, "arm": link2},
        joints=[joint],
    )

    # Must be parseable XML
    root = ET.fromstring(urdf_xml)
    assert root.tag == "robot"
    assert root.get("name") == "test_humanoid"

    # Both links present
    link_names = {el.get("name") for el in root.findall("link")}
    assert link_names == {"base", "arm"}

    # Joint with correct parent/child
    joints = root.findall("joint")
    assert len(joints) == 1
    j = joints[0]
    assert j.get("name") == "base_to_arm"
    assert j.get("type") == "revolute"
    assert j.find("parent").get("link") == "base"
    assert j.find("child").get("link") == "arm"


@pytest.mark.unit
def test_unknown_joint_type_falls_back_to_revolute() -> None:
    """Unknown joint type strings fall back to REVOLUTE rather than crash."""
    from humanoid_character_builder.core.model import GeneratedJoint
    from model_generation.core.types import JointType

    joint = GeneratedJoint(
        name="weird",
        joint_type="bogus_type",
        parent="a",
        child="b",
        origin_xyz=(0.0, 0.0, 0.0),
        origin_rpy=(0.0, 0.0, 0.0),
        axis=(0.0, 0.0, 1.0),
        limits=None,
        dynamics={},
    )
    _, canon_joints = to_canonical_lists({}, [joint])
    assert canon_joints[0].joint_type == JointType.REVOLUTE
