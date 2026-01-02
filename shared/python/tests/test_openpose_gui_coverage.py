import sys
from unittest.mock import patch

import pytest
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox

from shared.python.pose_estimation.openpose_gui import OpenPoseGUI


# Helper fixture to ensure QApplication exists
@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


@pytest.fixture
def gui(qapp, qtbot):
    window = OpenPoseGUI()
    qtbot.addWidget(window)
    return window


def test_initial_state(gui):
    assert gui.lbl_file.text() == "No file selected."
    assert not gui.btn_run.isEnabled()
    assert gui.progress.value() == 0


def test_load_video_cancel(gui):
    with patch.object(QFileDialog, "getOpenFileName", return_value=("", "")):
        gui.load_video()
        assert gui.lbl_file.text() == "No file selected."
        assert not gui.btn_run.isEnabled()


def test_load_video_success(gui):
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


def test_run_analysis(gui, qtbot):
    # Setup state
    gui.lbl_file.setText("/path/to/video.mp4")
    gui.btn_run.setEnabled(True)

    # Mock message box to prevent blocking
    with patch.object(QMessageBox, "information") as mock_msg:
        gui.run_analysis()

        assert not gui.btn_run.isEnabled()
        assert not gui.btn_load.isEnabled()
        assert "Starting analysis" in gui.log_area.toPlainText()

        # Verify timer started, then manually trigger completion to avoid slow waits

        assert gui.timer.isActive()
        gui.timer.stop()

        # Manually trigger completion
        gui.progress.setValue(100)
        gui.update_progress()

        assert "Analysis Complete!" in gui.log_area.toPlainText()
        assert gui.btn_run.isEnabled()
        mock_msg.assert_called_once()


def test_log(gui):
    gui.log("Test message")
    assert "Test message" in gui.log_area.toPlainText()
