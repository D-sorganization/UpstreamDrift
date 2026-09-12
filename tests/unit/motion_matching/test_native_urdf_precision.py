"""Native interchange must not round physical constants to six digits."""

import xml.etree.ElementTree as ET

import pytest

from src.shared.python.model_generation.builders.urdf_writer import URDFWriter
from src.shared.python.model_generation.core.types import (
    Inertia,
    Joint,
    JointType,
    Link,
    Origin,
)

pytestmark = pytest.mark.unit


def test_round_trip_retains_binary64_physical_values() -> None:
    value = 0.12345678901234567
    links = [
        Link(name="root", inertia=Inertia(0.0, 0.0, 0.0, mass=0.0)),
        Link(
            name="body",
            inertia=Inertia(
                value, value, value, mass=value, center_of_mass=(value, 0.0, 0.0)
            ),
        ),
    ]
    joint = Joint(
        name="fixed",
        joint_type=JointType.FIXED,
        parent="root",
        child="body",
        origin=Origin(xyz=(value, 0.0, 0.0), rpy=(0.0, value, 0.0)),
    )
    root = ET.fromstring(
        URDFWriter(numeric_precision=17).write("native", links, [joint])
    )
    assert float(root.find("link[@name='body']/inertial/mass").attrib["value"]) == value
    assert (
        float(root.find("link[@name='body']/inertial/inertia").attrib["ixx"]) == value
    )
    assert (
        float(root.find("link[@name='body']/inertial/origin").attrib["xyz"].split()[0])
        == value
    )
    assert float(root.find("joint/origin").attrib["rpy"].split()[1]) == value
    assert float(root.find("link[@name='root']/inertial/mass").attrib["value"]) == 0


@pytest.mark.parametrize("precision", [0, 18, True, 1.5])
def test_invalid_precision_rejected(precision: object) -> None:
    with pytest.raises(ValueError, match="precision"):
        URDFWriter(numeric_precision=precision)
