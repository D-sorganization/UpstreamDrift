"""Unit tests for Model Explorer GUI model loading and error propagation."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QMessageBox
import pytest

from src.tools.model_explorer.gui import MainWidget
from src.tools.model_explorer.main_window import URDFGeneratorWindow

pytestmark = [pytest.mark.unit, pytest.mark.ui]

_APP: QApplication | None = None


def _ensure_qapp() -> QApplication:
    global _APP
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    _APP = app
    return app


MINIMAL_VALID_URDF = """<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
</robot>
"""

CORRUPTED_BINARY_DATA = b"\x80\xff\xfe\x00\x01\x02\x03\x04\xff\xff"
CORRUPTED_OSIM_CONTENT = """<?xml version="1.0" encoding="UTF-8"?>
<OpenSimDocument Version="40000">
  <InvalidTag>
"""


@pytest.fixture
def valid_urdf_file(tmp_path: Path) -> Path:
    file = tmp_path / "valid_model.urdf"
    file.write_text(MINIMAL_VALID_URDF, encoding="utf-8")
    return file


@pytest.fixture
def corrupted_binary_file(tmp_path: Path) -> Path:
    file = tmp_path / "corrupted_model.urdf"
    file.write_bytes(CORRUPTED_BINARY_DATA)
    return file


@pytest.fixture
def corrupted_osim_file(tmp_path: Path) -> Path:
    file = tmp_path / "corrupted_model.osim"
    file.write_text(CORRUPTED_OSIM_CONTENT, encoding="utf-8")
    return file


def test_load_model_success_with_valid_urdf(valid_urdf_file: Path) -> None:
    _ensure_qapp()
    widget = MainWidget()

    success = widget.load_model(valid_urdf_file)

    assert success is True
    assert widget.current_file_path == valid_urdf_file


def test_load_model_accepts_str_and_path(valid_urdf_file: Path) -> None:
    _ensure_qapp()
    widget = MainWidget()

    assert widget.load_model(str(valid_urdf_file)) is True
    assert widget.current_file_path == valid_urdf_file


def test_load_model_returns_false_on_missing_file(tmp_path: Path) -> None:
    _ensure_qapp()
    widget = MainWidget()
    missing_file = tmp_path / "does_not_exist.urdf"

    with patch.object(QMessageBox, "warning") as mock_warning:
        success = widget.load_model(missing_file)

    assert success is False
    assert widget.current_file_path is None
    mock_warning.assert_called_once()


def test_load_model_returns_false_and_preserves_state_on_corrupted_binary_file(
    valid_urdf_file: Path,
    corrupted_binary_file: Path,
) -> None:
    _ensure_qapp()
    widget = MainWidget()

    # Load valid model first to establish prior state
    assert widget.load_model(valid_urdf_file) is True
    assert widget.current_file_path == valid_urdf_file

    with patch.object(QMessageBox, "warning") as mock_warning:
        # Load corrupted file should fail without raising
        success = widget.load_model(corrupted_binary_file)

    # State preserved: current_file_path should still be valid_urdf_file
    assert success is False
    assert widget.current_file_path == valid_urdf_file
    mock_warning.assert_called_once()


def test_load_model_returns_false_and_preserves_state_on_corrupted_osim_file(
    valid_urdf_file: Path,
    corrupted_osim_file: Path,
) -> None:
    _ensure_qapp()
    widget = MainWidget()

    # Load valid model first to establish prior state
    assert widget.load_model(valid_urdf_file) is True
    assert widget.current_file_path == valid_urdf_file

    with patch.object(QMessageBox, "warning") as mock_warning:
        # Load corrupted osim file should fail without raising
        success = widget.load_model(corrupted_osim_file)

    # State preserved: current_file_path should still be valid_urdf_file
    assert success is False
    assert widget.current_file_path == valid_urdf_file
    mock_warning.assert_called_once()


def test_load_model_returns_false_when_visualization_fails(
    valid_urdf_file: Path,
) -> None:
    _ensure_qapp()
    widget = MainWidget()

    with (
        patch.object(
            widget.visualization_widget,
            "update_visualization",
            side_effect=RuntimeError("OpenGL shader error"),
        ),
        patch.object(QMessageBox, "warning") as mock_warning,
    ):
        success = widget.load_model(valid_urdf_file)

    assert success is False
    assert widget.current_file_path is None
    mock_warning.assert_called_once()


def test_open_urdf_updates_status_bar_on_success(valid_urdf_file: Path) -> None:
    _ensure_qapp()
    widget = MainWidget()

    with patch(
        "PyQt6.QtWidgets.QFileDialog.getOpenFileName",
        return_value=(str(valid_urdf_file), "URDF Files (*.urdf)"),
    ):
        widget.open_urdf()

    assert widget.status_bar.currentMessage() == f"Opened: {valid_urdf_file}"
    assert widget.current_file_path == valid_urdf_file


def test_open_urdf_does_not_overwrite_status_bar_on_failure(tmp_path: Path) -> None:
    _ensure_qapp()
    widget = MainWidget()
    widget.status_bar.showMessage("Initial Status")
    missing_file = tmp_path / "nonexistent.urdf"

    with (
        patch(
            "PyQt6.QtWidgets.QFileDialog.getOpenFileName",
            return_value=(str(missing_file), "URDF Files (*.urdf)"),
        ),
        patch.object(QMessageBox, "warning"),
    ):
        widget.open_urdf()

    # Status bar should retain previous message and not show "Opened: ..."
    assert widget.status_bar.currentMessage() == "Initial Status"
    assert widget.current_file_path is None


def test_load_from_library_does_not_overwrite_status_bar_on_load_failure() -> None:
    _ensure_qapp()
    widget = MainWidget()
    widget.status_bar.showMessage("Ready")

    mock_dialog = MagicMock()
    mock_dialog.exec.return_value = True
    mock_dialog.get_selected_model.return_value = ("golf_clubs", "custom_driver")

    with (
        patch(
            "src.tools.model_explorer.model_loader_dialog.ModelLoaderDialog",
            return_value=mock_dialog,
        ),
        patch(
            "src.tools.model_explorer.model_library.ModelLibrary.generate_golf_club_urdf",
            return_value=Path("/invalid/path/club.urdf"),
        ),
        patch.object(widget, "load_model", return_value=False),
        patch.object(QMessageBox, "warning"),
    ):
        widget.load_from_library()

    assert widget.status_bar.currentMessage() == "Ready"


def test_load_default_model_guards_status_bar_on_load_failure() -> None:
    _ensure_qapp()
    widget = MainWidget()
    widget.status_bar.showMessage("Ready")

    with (
        patch(
            "src.tools.model_explorer.model_library.ModelLibrary.get_human_model",
            return_value=Path("/invalid/model.urdf"),
        ),
        patch("pathlib.Path.exists", return_value=True),
        patch.object(widget, "load_model", return_value=False),
    ):
        widget.load_default_model()

    assert widget.status_bar.currentMessage() == "Ready"


def test_urdf_generator_window_delegates_load_model(valid_urdf_file: Path) -> None:
    _ensure_qapp()
    window = URDFGeneratorWindow()

    assert window.load_model(valid_urdf_file) is True
    assert window.current_file_path == valid_urdf_file

    with patch.object(QMessageBox, "warning"):
        assert window.load_model(Path("/invalid/file.urdf")) is False


def test_load_urdf_file_backward_compatibility_alias(valid_urdf_file: Path) -> None:
    _ensure_qapp()
    widget = MainWidget()

    assert widget._load_urdf_file(valid_urdf_file) is True
    assert widget.current_file_path == valid_urdf_file
