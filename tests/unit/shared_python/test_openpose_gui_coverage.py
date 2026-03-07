from __future__ import annotations

import sys
from collections.abc import Generator
from unittest.mock import patch

import pytest

# Skip entire module if PyQt6 GUI libraries are not available
try:
    from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox

    PYQT6_AVAILABLE = True
except (ImportError, OSError):
    PYQT6_AVAILABLE = False
    QApplication = None  # type: ignore[misc, assignment]
    QFileDialog = None  # type: ignore[misc, assignment]
    QMessageBox = None  # type: ignore[misc, assignment]

pytestmark = pytest.mark.skipif(
    not PYQT6_AVAILABLE, reason="PyQt6 GUI libraries not available"
)

if PYQT6_AVAILABLE:
    from shared.python.pose_estimation.openpose_gui import OpenPoseGUI


# Helper fixture to ensure QApplication exists
@pytest.fixture(scope="session")
def qapp() -> Generator[QApplication, None, None]:
    """Provide the QApplication instance for the session."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app  # type: ignore[misc]


@pytest.fixture
def gui(qapp, qtbot) -> OpenPoseGUI:
    """Create an OpenPoseGUI instance."""
    window = OpenPoseGUI()
    qtbot.addWidget(window)
    return window


def test_initial_state(gui) -> None:
    """Test OpenPoseGUI initial widget state."""
    assert gui.lbl_file.text() == "No file selected."
    assert not gui.btn_run.isEnabled()
    assert gui.progress.value() == 0


def test_load_video_cancel(gui) -> None:
    """Test loading video when user cancels file dialog."""
    with patch.object(QFileDialog, "getOpenFileName", return_value=("", "")):
        gui.load_video()
        assert gui.lbl_file.text() == "No file selected."
        assert not gui.btn_run.isEnabled()


def test_load_video_success(gui) -> None:
    """Test successful video file loading."""
    test_file = "/path/to/video.mp4"
    with patch.object(
        QFileDialog,
        "getOpenFileName",
        return_value=(test_file, "Video Files (*.mp4 *.avi *.mov)"),
    ):
        gui.load_video()
        assert gui.lbl_file.text() == test_file
        assert gui.btn_run.isEnabled()
        assert "Loaded video" in gui.log_area.toPlainText()


def test_run_analysis(gui, qtbot) -> None:
    """Test running analysis workflow."""
    from unittest.mock import MagicMock

    # Setup state — must set _video_path directly (lbl_file.setText alone doesn't set it)
    gui._video_path = "/path/to/video.mp4"
    gui.lbl_file.setText("/path/to/video.mp4")
    gui.btn_run.setEnabled(True)

    # Mock the _AnalysisWorker so no real thread starts
    mock_worker = MagicMock()
    mock_worker.progress = MagicMock()
    mock_worker.finished = MagicMock()
    mock_worker.error = MagicMock()

    with patch(
        "shared.python.pose_estimation.openpose_gui._AnalysisWorker",
        return_value=mock_worker,
    ):
        gui.run_analysis()

        assert not gui.btn_run.isEnabled()
        assert not gui.btn_load.isEnabled()
        assert "Starting OpenPose analysis" in gui.log_area.toPlainText()

        # Simulate worker completion: directly call _on_finished
        with patch(
            "shared.python.pose_estimation.openpose_gui.QMessageBox"
        ) as mock_msg_cls:
            gui._on_finished([])  # Trigger completion with empty results
            assert "Analysis complete!" in gui.log_area.toPlainText()
            assert gui.btn_run.isEnabled()
            mock_msg_cls.information.assert_called_once()


def test_log(gui) -> None:
    """Test log message appending."""
    gui.log("Test message")
    assert "Test message" in gui.log_area.toPlainText()
