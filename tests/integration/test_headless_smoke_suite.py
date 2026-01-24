"""Headless smoke test suite for the C3D Viewer and key GUI components.

This suite ensures that the application can initialize and perform basic operations
without a physical display, suitable for CI/CD environments.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.shared.python.path_utils import get_simscape_model_path, setup_import_paths

# Setup import paths including Simscape model
setup_import_paths(additional_paths=[get_simscape_model_path()])


# Conditional import to handle potential import errors gracefully during collection
try:
    from apps.c3d_viewer import C3DViewerMainWindow
    from apps.core.models import C3DDataModel
except ImportError:
    C3DViewerMainWindow = None
    C3DDataModel = None


@pytest.fixture
def mock_loader_thread():
    """Mock the C3DLoaderThread to prevent actual thread execution."""
    with patch("apps.c3d_viewer.C3DLoaderThread") as MockThread:
        mock_instance = MockThread.return_value
        # Setup signals
        mock_instance.loaded = MagicMock()
        mock_instance.failed = MagicMock()
        mock_instance.progress = MagicMock()
        yield MockThread


@pytest.mark.skipif(
    C3DViewerMainWindow is None, reason="Could not import C3DViewerMainWindow"
)
class TestHeadlessSuite:
    """Headless integration tests for C3D Viewer."""

    def test_mainwindow_startup(self, qtbot, mock_loader_thread):
        """Verify that the main window initializes correctly."""
        window = C3DViewerMainWindow()
        qtbot.addWidget(window)

        assert window.windowTitle() == "C3D Motion Analysis Viewer"
        assert window.isVisible() is False  # Should be invisible by default in test

        # Verify tabs
        tab_widget = window.centralWidget()
        assert tab_widget.count() > 0

        # Check for specific expected tabs
        tab_texts = [tab_widget.tabText(i) for i in range(tab_widget.count())]
        assert "3D Viewer" in tab_texts
        assert "Overview" in tab_texts
        # "Advanced Plots" might be conditional or pending, but checking core tabs

    def test_file_loading_trigger(self, qtbot, mock_loader_thread):
        """Verify that opening a file triggers the loader thread."""
        window = C3DViewerMainWindow()
        qtbot.addWidget(window)

        # Mock the QFileDialog to return a dummy path
        with patch("PyQt6.QtWidgets.QFileDialog.getOpenFileName") as mock_dialog:
            mock_dialog.return_value = ("test_capture.c3d", "C3D Files (*.c3d)")

            # Trigger file open
            # We need to mock validate_path inside the method or assume it passes for "test_capture.c3d"
            # However, open_c3d_file calls validate_path which uses Path().resolve()
            # If "test_capture.c3d" is relative, it becomes absolute.

            # Since open_c3d_file does local import of validate_path, we might need to rely on
            # the fact that it resolves the path.

            with patch(
                "src.shared.python.security_utils.validate_path"
            ) as mock_validate:
                # Mock validate to return the input path as absolute
                abs_path = os.path.abspath("test_capture.c3d")
                mock_validate.return_value = abs_path

                window.open_c3d_file()

                # Check that loader thread was started with expected absolute path
                mock_loader_thread.assert_called_with(str(abs_path))
                mock_loader_thread.return_value.start.assert_called_once()

            # Verify status bar indicates loading
            # The status bar message uses os.path.basename, so it should still say "test_capture.c3d"
            assert "Loading test_capture.c3d" in window.statusBar().currentMessage()

    def test_successful_load_ui_update(self, qtbot, mock_loader_thread):
        """Verify UI updates upon successful data load."""
        window = C3DViewerMainWindow()
        qtbot.addWidget(window)

        # Simulate loader thread emitting 'loaded' signal
        mock_data = MagicMock(spec=C3DDataModel)
        mock_data.filepath = "test.c3d"  # Need filepath for status message
        mock_data.metadata = {"File": "test.c3d", "Points": "10"}
        mock_data.markers = {}
        mock_data.analog = {}
        mock_data.marker_names.return_value = []
        mock_data.analog_names.return_value = []
        mock_data.point_time = None
        mock_data.analog_time = None

        # Call the slot directly to simulate signal emission
        window._on_load_success(mock_data)

        assert window.model == mock_data
        # Window title isn't updated in the current implementation, just the status bar and internal state
        # (Based on reading c3d_viewer.py lines 359-366)

        # Verify status bar updated
        assert "Loaded test.c3d successfully" in window.statusBar().currentMessage()

    def test_failed_load_ui_update(self, qtbot, mock_loader_thread):
        """Verify UI updates upon load failure."""
        window = C3DViewerMainWindow()
        qtbot.addWidget(window)

        # Mock critical message box failure
        # QMessageBox.critical blocks, so we must mock it
        with patch("PyQt6.QtWidgets.QMessageBox.critical") as mock_critical:
            window._on_load_failure("Corrupt file")

            assert window.model is None
            mock_critical.assert_called_once()
            assert "Corrupt file" in mock_critical.call_args[0][2]
