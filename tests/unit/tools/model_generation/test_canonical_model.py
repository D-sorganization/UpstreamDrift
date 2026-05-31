"""Tests for the canonical model representation."""

from __future__ import annotations

import dataclasses
import json
import xml.etree.ElementTree as ET

import pytest
from src.shared.python.model_generation.canonical_model import (
    CanonicalGeometry,
    CanonicalInertia,
    CanonicalJoint,
    CanonicalJointLimits,
    CanonicalLink,
    CanonicalMaterial,
    CanonicalModel,
)
from src.shared.python.model_generation.core.types import (
    Geometry,
    GeometryType,
    Inertia,
    Joint,
    JointType,
    Link,
)


def _base_inertia() -> CanonicalInertia:
    return CanonicalInertia(mass=1.0, ixx=0.2, iyy=0.2, izz=0.2)


def _simple_model() -> CanonicalModel:
    return CanonicalModel(
        name="simple_arm",
        links=(
            CanonicalLink(
                name="tool",
                inertia=_base_inertia(),
                visual_geometry=CanonicalGeometry(
                    geometry_type="sphere", dimensions=(0.05,)
                ),
            ),
            CanonicalLink(
                name="base",
                inertia=_base_inertia(),
                visual_geometry=CanonicalGeometry(
                    geometry_type="box", dimensions=(0.2, 0.2, 0.1)
                ),
                visual_material=CanonicalMaterial(
                    name="matte", color=(0.1, 0.2, 0.3, 1.0)
                ),
            ),
        ),
        joints=(
            CanonicalJoint(
                name="base_to_tool",
                joint_type="revolute",
                parent="base",
                child="tool",
                axis=(0.0, 0.0, 1.0),
                limits=CanonicalJointLimits(lower=-1.0, upper=1.0),
            ),
        ),
        metadata={"source": "unit-test"},
    )


def test_canonical_model_is_immutable_and_sorts_components() -> None:
    model = CanonicalModel(
        name="immutable",
        links=(
            CanonicalLink(
                name="base",
                inertia=_base_inertia(),
                visual_geometry=CanonicalGeometry(
                    geometry_type=GeometryType.BOX, dimensions=(0.2, 0.2, 0.1)
                ),
            ),
        ),
        metadata={"nested": {"source": "unit-test"}},
    )

    assert [link.name for link in model.links] == ["base"]
    assert model.links[0].visual_geometry is not None
    assert model.links[0].visual_geometry.geometry_type == "box"
    assert model.metadata["nested"]["source"] == "unit-test"
    with pytest.raises(dataclasses.FrozenInstanceError):
        model.name = "other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        model.metadata["nested"]["source"] = "other"  # type: ignore[index]


def test_canonical_model_json_and_hash_are_stable() -> None:
    model = _simple_model()
    reversed_model = CanonicalModel(
        name=model.name,
        links=tuple(reversed(model.links)),
        joints=tuple(reversed(model.joints)),
        metadata={"source": "unit-test"},
    )

    payload = json.loads(model.to_json())

    assert payload["schema_version"] == 1
    assert model.to_json() == reversed_model.to_json()
    assert model.stable_hash() == reversed_model.stable_hash()
    assert len(model.stable_hash()) == 64


def test_canonical_model_validates_geometry_before_export() -> None:
    model = CanonicalModel(
        name="bad_geometry",
        links=(
            CanonicalLink(
                name="base",
                inertia=_base_inertia(),
                visual_geometry=CanonicalGeometry(
                    geometry_type="box", dimensions=(1.0,)
                ),
            ),
        ),
    )

    result = model.validate()

    assert not result.is_valid
    assert "CANONICAL_GEOMETRY_DIMENSIONS" in result.get_error_messages()[0]
    with pytest.raises(ValueError, match="CANONICAL_GEOMETRY_DIMENSIONS"):
        model.to_urdf()


def test_canonical_model_roundtrips_existing_core_types() -> None:
    links = [
        Link(
            name="base",
            inertia=Inertia(ixx=0.2, iyy=0.2, izz=0.2, mass=1.0),
            visual_geometry=Geometry.box(0.1, 0.2, 0.3),
        ),
        Link(name="tool", inertia=Inertia(ixx=0.2, iyy=0.2, izz=0.2, mass=1.0)),
    ]
    joints = [
        Joint(
            name="base_to_tool",
            joint_type=JointType.FIXED,
            parent="base",
            child="tool",
        )
    ]

    model = CanonicalModel.from_core(name="core_model", links=links, joints=joints)
    core_links, core_joints = model.to_core()

    assert [link.name for link in core_links] == ["base", "tool"]
    assert core_joints[0].joint_type == JointType.FIXED


def test_canonical_model_exports_valid_urdf_with_existing_writer() -> None:
    xml = _simple_model().to_urdf()
    root = ET.fromstring(xml)

    assert root.tag == "robot"
    assert root.attrib["name"] == "simple_arm"
    assert root.find("./link[@name='base']") is not None
    assert root.find("./joint[@name='base_to_tool']") is not None
