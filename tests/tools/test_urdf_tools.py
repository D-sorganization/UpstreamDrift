import os
import sys

import pytest
from src.shared.python.data_io.common_utils import get_shared_urdf_path
from src.shared.python.engine_core.engine_availability import PYQT6_AVAILABLE

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


if __name__ == "__main__":
    pytest.main([__file__])
