import sys
from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtWidgets import QApplication

from launchers.golf_launcher import EnvironmentDialog

# Mocking modules that might cause issues in headless environment
sys.modules["shared.python.engine_manager"] = MagicMock()
sys.modules["shared.python.model_registry"] = MagicMock()
sys.modules["shared.python.secure_subprocess"] = MagicMock()


@pytest.fixture
def app():
    """Create a Qt application instance."""
    app = QApplication.instance()
    if not app:
        app = QApplication([])
    return app


@pytest.fixture
def dialog(app):
    """Create the EnvironmentDialog instance with mocked parent."""
    with patch("launchers.golf_launcher.DockerBuildThread"):
        dlg = EnvironmentDialog(None)
        yield dlg
        dlg.close()


def test_build_button_feedback(dialog):
    """Test that the build button text changes during build process."""

    # Initial state
    initial_text = dialog.btn_build.text()
    assert initial_text == "Build Environment"
    assert dialog.btn_build.isEnabled() is True

    # Start build
    with patch("launchers.golf_launcher.DockerBuildThread") as MockThread:
        _ = MockThread.return_value

        dialog.start_build()

        # Verify button state changed
        assert dialog.btn_build.text() == "Building..."
        assert dialog.btn_build.isEnabled() is False

        # Finish build (success)
        with patch("PyQt6.QtWidgets.QMessageBox.exec"):
            dialog.build_finished(True, "Success")

        # Verify button state restored
        assert dialog.btn_build.text() == initial_text
        assert dialog.btn_build.isEnabled() is True


def test_build_button_feedback_failure(dialog):
    """Test that the build button text restores even on failure."""

    # Start build
    with patch("launchers.golf_launcher.DockerBuildThread"):
        dialog.start_build()

        # Verify button state changed
        assert dialog.btn_build.text() == "Building..."

        # Finish build (failure)
        with patch("PyQt6.QtWidgets.QMessageBox.exec"):
            dialog.build_finished(False, "Failed")

        # Verify button state restored
        assert dialog.btn_build.text() == "Build Environment"
        assert dialog.btn_build.isEnabled() is True
