"""
Integration test for verifying UpstreamDriftLauncher logic regarding X11 environment flags.
Ensure that selecting 'Live Visualization' correctly sets the necessary
OSMesa/GLFW/X11 environment variables, specifically testing for the
presence of LIBGL_ALWAYS_INDIRECT which was identified as a critical regression.
"""

import sys
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# These tests verify Windows-specific X11 forwarding behavior (VcXSrv + DISPLAY).
# They must be skipped on non-Windows platforms where the Docker command format differs.
pytestmark = pytest.mark.skipif(
    sys.platform != "win32",
    reason="X11 forwarding tests are Windows-specific (DISPLAY=host.docker.internal:0)",
)


# Dummy Qt Logic for Headless Testing
class MockQCheckBox:
    def __init__(self, checked=False):
        self._checked = checked

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, val) -> None:
        self._checked = val


class MockModel:
    def __init__(self, model_type="custom_humanoid"):
        self.type = model_type
        self.id = "test_model"
        self.name = f"test_{model_type}"


@pytest.fixture
def mocked_launcher() -> Generator[Any, None, None]:
    """Import upstream_drift_launcher with Qt mocks."""
    mock_modules = {
        "PyQt6": MagicMock(),
        "PyQt6.QtCore": MagicMock(),
        "PyQt6.QtGui": MagicMock(),
        "PyQt6.QtWidgets": MagicMock(),
    }
    mock_modules["PyQt6.QtWidgets"].QMainWindow = object
    mock_modules["PyQt6.QtWidgets"].QCheckBox = MockQCheckBox

    with patch.dict(sys.modules, mock_modules):
        import src.launchers.upstream_drift_launcher

        # Patch the class to avoid __init__ doing GUI stuff
        class TestLauncher(src.launchers.upstream_drift_launcher.UpstreamDriftLauncher):
            def __init__(self):
                self.chk_live: MockQCheckBox = MockQCheckBox(checked=True)  # type: ignore[assignment]
                self.chk_gpu: MockQCheckBox = MockQCheckBox(checked=False)  # type: ignore[assignment]
                self.model_cards = {}
                self.docker_launcher = MagicMock(
                    spec=["check_image_exists", "build_image", "run_container"]
                )
                self.process_manager = MagicMock(
                    spec=["start_process", "stop_process", "is_running"]
                )
                self.lbl_status = MagicMock(spec=["setText", "setStyleSheet"])
                self.toast_manager = None
                self.show_toast = MagicMock(spec=["__call__"])

            # Override _launch_docker_container to just return the command checks
            # or we can test the actual method if we mock start_meshcat etc.
            def _start_meshcat_browser(self, port) -> None:
                pass

        yield TestLauncher
