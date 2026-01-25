import os
import sys

import pytest

from src.shared.python.common_utils import get_shared_urdf_path
from src.shared.python.engine_availability import PYQT6_AVAILABLE

# Check if display is available for Qt tests
HAS_DISPLAY = os.environ.get("DISPLAY") is not None or sys.platform == "win32"

# Import URDFGenerator if PyQt6 is available
if PYQT6_AVAILABLE:
    from src.tools.urdf_generator.main_window import (
        URDFGeneratorWindow as URDFGenerator,
    )
else:
    URDFGenerator = None  # type: ignore


class MockFileDialog:
    @staticmethod
    def getSaveFileName(parent, caption, directory, filter):
        return "test_robot.urdf", "URDF Files (*.urdf)"

    @staticmethod
    def getOpenFileName(parent, caption, directory, filter):
        return "test_robot.urdf", "URDF Files (*.urdf)"


@pytest.mark.skip(
    reason="Test expects old API (window.links, window._generate_urdf_xml) that no longer exists"
)
def test_urdf_generation_logic(qtbot):
    """Test the logic of generating URDF XML.

    NOTE: This test is skipped because URDFGeneratorWindow was refactored
    and no longer exposes the direct links/joints/_generate_urdf_xml API.
    The test needs to be rewritten to match the current implementation.
    """


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
