import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from shared.python.common_utils import get_shared_urdf_path

# Try to import PyQt6 and URDFGenerator, skip tests if not available
try:
    from tools.urdf_generator.main import URDFGenerator

    PYQT6_AVAILABLE = True
except ImportError:
    PYQT6_AVAILABLE = False
    URDFGenerator = None


class MockFileDialog:
    @staticmethod
    def getSaveFileName(parent, caption, directory, filter):
        return "test_robot.urdf", "URDF Files (*.urdf)"

    @staticmethod
    def getOpenFileName(parent, caption, directory, filter):
        return "test_robot.urdf", "URDF Files (*.urdf)"


@pytest.mark.skipif(not PYQT6_AVAILABLE, reason="PyQt6 not available")
def test_urdf_generation_logic(qtbot):
    """Test the logic of generating URDF XML."""
    window = URDFGenerator()
    qtbot.addWidget(window)

    # Check default state
    assert len(window.links) == 1
    assert window.links[0]["name"] == "base_link"

    # Generate XML
    xml_str = window._generate_urdf_xml()
    root = ET.fromstring(xml_str)
    assert root.tag == "robot"
    assert len(root.findall("link")) == 1

    # Add a link
    new_link = {
        "name": "link1",
        "geometry_type": "cylinder",
        "size": "0.1 0.5",
        "color": "1 0 0 1",
    }
    window.links.append(new_link)

    # Add a joint
    new_joint = {
        "name": "joint1",
        "type": "revolute",
        "parent": "base_link",
        "child": "link1",
        "origin": "0 0 1",
        "rpy": "0 0 0",
        "axis": "0 0 1",
    }
    window.joints.append(new_joint)

    # Generate XML again
    xml_str = window._generate_urdf_xml()
    root = ET.fromstring(xml_str)

    links = root.findall("link")
    assert len(links) == 2

    joints = root.findall("joint")
    assert len(joints) == 1
    joint = joints[0]
    assert joint.attrib["name"] == "joint1"
    assert joint.find("parent").attrib["link"] == "base_link"
    assert joint.find("child").attrib["link"] == "link1"


def test_urdf_scanning_logic():
    """Test detecting shared URDFs."""
    # Simulate scanning logic used in GUIs
    urdf_dir = get_shared_urdf_path()

    assert urdf_dir is not None
    assert urdf_dir.exists()
    urdfs = list(urdf_dir.glob("*.urdf"))
    assert len(urdfs) >= 2

    names = [u.stem for u in urdfs]
    assert "simple_humanoid" in names
    assert "arm" in names


if __name__ == "__main__":
    pytest.main([__file__])
