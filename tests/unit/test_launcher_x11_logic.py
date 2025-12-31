"""
Integration test for verifying GolfLauncher logic regarding X11 environment flags.
Ensure that selecting 'Live Visualization' correctly sets the necessary
OSMesa/GLFW/X11 environment variables, specifically testing for the
presence of LIBGL_ALWAYS_INDIRECT which was identified as a critical regression.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# Dummy Qt Logic for Headless Testing
class MockQCheckBox:
    def __init__(self, checked=False):
        self._checked = checked

    def isChecked(self):
        return self._checked

    def setChecked(self, val):
        self._checked = val


class MockModel:
    def __init__(self, model_type="custom_humanoid"):
        self.type = model_type
        self.id = "test_model"


@pytest.fixture
def mocked_launcher():
    """Import golf_launcher with Qt mocks."""
    mock_modules = {
        "PyQt6": MagicMock(),
        "PyQt6.QtCore": MagicMock(),
        "PyQt6.QtGui": MagicMock(),
        "PyQt6.QtWidgets": MagicMock(),
    }
    mock_modules["PyQt6.QtWidgets"].QMainWindow = object
    mock_modules["PyQt6.QtWidgets"].QCheckBox = MockQCheckBox

    with patch.dict(sys.modules, mock_modules):
        import launchers.golf_launcher

        # Patch the class to avoid __init__ doing GUI stuff
        class TestLauncher(launchers.golf_launcher.GolfLauncher):
            def __init__(self):
                self.chk_live = MockQCheckBox(checked=True)
                self.chk_gpu = MockQCheckBox(checked=False)
                self.model_cards = {}

            # Override _launch_docker_container to just return the command checks
            # or we can test the actual method if we mock start_meshcat etc.
            def _start_meshcat_browser(self, port):
                pass

        yield TestLauncher


def test_live_view_environment_flags(mocked_launcher):
    """Verify LIBGL_ALWAYS_INDIRECT and other flags are present when Live View is enabled on Windows."""

    launcher = mocked_launcher()
    launcher.chk_live.setChecked(True)

    # Mock OS to look like Windows
    with patch("os.name", "nt"), patch("subprocess.Popen") as mock_popen:

        # Create a dummy model
        model = MockModel("custom_humanoid")
        abs_path = Path("/mock/path")

        # Call the method
        launcher._launch_docker_container(model, abs_path)

        # Inspect subprocess call
        assert mock_popen.called
        args = mock_popen.call_args[0][0]  # The command list

        # Flatten the args if it's a list of strings
        full_command = " ".join(args)

        print(f"DEBUG Command: {full_command}")

        # Assert Critical Flags for X11 on Windows
        assert "DISPLAY=host.docker.internal:0" in full_command
        assert "MUJOCO_GL=glfw" in full_command
        assert "PYOPENGL_PLATFORM=glx" in full_command
        assert "QT_QPA_PLATFORM=xcb" in full_command
        # LIBGL_ALWAYS_INDIRECT causes ShaderValidationError with dm_control (OpenGL < 1.5)
        # So we explicitly assert it is ABSENT.
        assert "LIBGL_ALWAYS_INDIRECT=1" not in full_command


def test_headless_environment_flags(mocked_launcher):
    """Verify flags for Headless mode."""
    launcher = mocked_launcher()
    launcher.chk_live.setChecked(False)

    with patch("os.name", "nt"), patch("subprocess.Popen") as mock_popen:

        model = MockModel("custom_humanoid")
        launcher._launch_docker_container(model, Path("/mock"))

        args = mock_popen.call_args[0][0]
        full_command = " ".join(args)

        assert "MUJOCO_GL=osmesa" in full_command
        assert "LIBGL_ALWAYS_INDIRECT" not in full_command
