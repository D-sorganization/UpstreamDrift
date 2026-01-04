
import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from PyQt6 import QtWidgets

# Use offscreen platform for headless testing
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from apps.c3d_viewer import C3DViewerMainWindow
from apps.core.models import C3DDataModel

@pytest.fixture
def app(qtbot):
    """Fixture to provide the main window."""
    window = C3DViewerMainWindow()
    qtbot.addWidget(window)
    return window

def test_viewer_startup(app):
    """Smoke test: Verify the application starts up without crashing."""
    assert app.windowTitle() == "C3D Motion Analysis Viewer"
    assert app.tabs.count() == 5  # Ensure all 5 tabs are created

def test_load_model_ui_update(app, qtbot):
    """Verify UI updates when a model is loaded."""
    # Mock data model
    model = C3DDataModel(
        filepath="/tmp/test.c3d",
        markers={},
        analog={},
        point_rate=100.0,
        analog_rate=1000.0,
        metadata={"Test": "Data"},
    )
    
    # Directly trigger the success handler to bypass threading/file IO
    app._on_load_success(model)
    
    # Check if UI updated (e.g., window title or internal state)
    # The file label in Overview tab should be updated
    assert "test.c3d" in app.overview_tab.label_file.text()
    assert app.model is model

@patch("apps.c3d_viewer.C3DLoaderThread")
def test_open_file_dialog_cancel(mock_loader, app, monkeypatch):
    """Test that cancelling the file dialog does nothing."""
    # Mock file dialog to return empty
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", lambda *args: ("", ""))
    
    app.open_c3d_file()
    
    # Loader thread should not be started
    mock_loader.assert_not_called()
