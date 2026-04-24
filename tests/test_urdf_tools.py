import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from src.shared.python.data_io.common_utils import get_shared_urdf_path
from src.shared.python.engine_core.engine_availability import PYQT6_AVAILABLE

_REPO_ROOT = Path(__file__).parent.parent
_SIMPLE_HUMANOID = _REPO_ROOT / "src/shared/urdf/simple_humanoid.urdf"
_GOLFER_URDF = (
    _REPO_ROOT / "src/engines/physics_engines/pinocchio/models/generated/golfer.urdf"
)

# Check if display is available for Qt tests
HAS_DISPLAY = os.environ.get("DISPLAY") is not None or sys.platform == "win32"

# Import URDFGenerator if PyQt6 is available
if PYQT6_AVAILABLE:
    try:
        from src.tools.model_explorer.main_window import (
            URDFGeneratorWindow as URDFGenerator,
        )
    except (ImportError, OSError):
        # QtOpenGLWidgets DLL may fail to load in CI or headless environments
        URDFGenerator = None  # type: ignore[assignment, misc]
else:
    URDFGenerator = None  # type: ignore[assignment, misc]


class MockFileDialog:
    @staticmethod
    def getSaveFileName(parent, caption, directory, filter):
        return "test_robot.urdf", "URDF Files (*.urdf)"

    @staticmethod
    def getOpenFileName(parent, caption, directory, filter):
        return "test_robot.urdf", "URDF Files (*.urdf)"


@pytest.mark.xfail(
    strict=False, reason="Shared URDF assets not provisioned in CI (#1949)"
)
def test_urdf_scanning_logic() -> None:
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


class TestSimpleHumanoidAnthropometry:
    """Verify simple_humanoid.urdf uses de Leva 1996 anthropometry-derived values."""

    def _get_link_mass(self, root: ET.Element, link_name: str) -> float:
        for link in root.iter("link"):
            if link.get("name") == link_name:
                inertial = link.find("inertial")
                if inertial is None:
                    pytest.fail(f"Link {link_name!r} has no <inertial>")
                mass_el = inertial.find("mass")
                if mass_el is None:
                    pytest.fail(f"Link {link_name!r} has no <mass>")
                return float(mass_el.get("value", "0"))
        pytest.fail(f"Link {link_name!r} not found in URDF")

    def _get_link_inertia(self, root: ET.Element, link_name: str) -> dict:
        for link in root.iter("link"):
            if link.get("name") == link_name:
                inertial = link.find("inertial")
                assert inertial is not None
                inertia_el = inertial.find("inertia")
                assert inertia_el is not None
                return {k: float(inertia_el.get(k, "0")) for k in ("ixx", "iyy", "izz")}
        pytest.fail(f"Link {link_name!r} not found")

    @pytest.fixture()
    def urdf_root(self):
        assert _SIMPLE_HUMANOID.exists(), f"Missing {_SIMPLE_HUMANOID}"
        return ET.parse(_SIMPLE_HUMANOID).getroot()

    def test_torso_mass_is_de_leva_1996(self, urdf_root):
        mass = self._get_link_mass(urdf_root, "torso")
        assert (
            abs(mass - 30.42) < 0.1
        ), f"torso mass {mass} not near de Leva 1996 value 30.42 kg"

    def test_head_mass_is_de_leva_1996(self, urdf_root):
        mass = self._get_link_mass(urdf_root, "head")
        assert (
            abs(mass - 4.86) < 0.05
        ), f"head mass {mass} not near de Leva 1996 value 4.86 kg"

    def test_upper_arm_mass_is_de_leva_1996(self, urdf_root):
        for link_name in ("right_upper_arm", "left_upper_arm"):
            mass = self._get_link_mass(urdf_root, link_name)
            assert abs(mass - 1.96) < 0.05, f"{link_name} mass {mass} not near 1.96 kg"

    def test_upper_leg_mass_is_de_leva_1996(self, urdf_root):
        for link_name in ("right_upper_leg", "left_upper_leg"):
            mass = self._get_link_mass(urdf_root, link_name)
            assert abs(mass - 7.0) < 0.1, f"{link_name} mass {mass} not near 7.0 kg"

    def test_torso_inertia_consistent_with_geometry(self, urdf_root):
        inertia = self._get_link_inertia(urdf_root, "torso")
        # Box 0.2x0.4x0.6 m at 30.42 kg: ixx=m(b²+c²)/12, iyy=m(a²+c²)/12, izz=m(a²+b²)/12
        assert abs(inertia["ixx"] - 1.318) < 0.01
        assert abs(inertia["iyy"] - 1.014) < 0.01
        assert abs(inertia["izz"] - 0.507) < 0.01

    def test_inertia_values_are_not_round_numbers(self, urdf_root):
        for link_name in ("torso", "head", "right_upper_arm", "right_upper_leg"):
            inertia = self._get_link_inertia(urdf_root, link_name)
            for axis, val in inertia.items():
                msg = f"{link_name}.{axis}={val} is a suspiciously round number"
                assert val != round(val, 1) or val < 0.01, msg


class TestGolferUrdfRightHandGrip:
    """Verify golfer.urdf has a right-hand grip attachment joint/link."""

    @pytest.fixture()
    def urdf_root(self):
        assert _GOLFER_URDF.exists(), f"Missing {_GOLFER_URDF}"
        return ET.parse(_GOLFER_URDF).getroot()

    def test_right_grip_attachment_link_exists(self, urdf_root):
        names = {el.get("name") for el in urdf_root.iter("link")}
        assert (
            "right_grip_attachment" in names
        ), "golfer.urdf missing right_grip_attachment link"

    def test_right_grip_joint_connects_to_hand_right(self, urdf_root):
        for joint in urdf_root.iter("joint"):
            if joint.get("name") == "hand_right_to_right_grip_attachment":
                parent = joint.find("parent")
                child = joint.find("child")
                assert parent is not None and parent.get("link") == "hand_right"
                assert (
                    child is not None and child.get("link") == "right_grip_attachment"
                )
                return
        pytest.fail(
            "joint hand_right_to_right_grip_attachment not found in golfer.urdf"
        )

    def test_right_grip_joint_is_fixed(self, urdf_root):
        for joint in urdf_root.iter("joint"):
            if joint.get("name") == "hand_right_to_right_grip_attachment":
                assert joint.get("type") == "fixed"
                return
        pytest.fail("joint hand_right_to_right_grip_attachment not found")

    def test_left_hand_club_shaft_joint_still_present(self, urdf_root):
        names = {el.get("name") for el in urdf_root.iter("joint")}
        assert (
            "hand_left_to_club_shaft" in names
        ), "left-hand grip was accidentally removed"


if __name__ == "__main__":
    pytest.main([__file__])
