"""Tests for Model Explorer per-engine export and viewer error handling (#12997)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtWidgets import QApplication, QMessageBox

from src.tools.model_explorer._mujoco_viewer_backend import MUJOCO_AVAILABLE
from src.tools.model_explorer.gui import MainWidget
from src.tools.model_explorer.mujoco_viewer import MuJoCoViewerWidget
from src.tools.model_explorer.segment_manager import SegmentManager
from tests.tools.model_explorer._fixtures import SIMPLE_URDF

pytestmark = [pytest.mark.unit, pytest.mark.ui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


_APP: QApplication | None = None


def _ensure_qapp() -> QApplication:
    global _APP
    app = QApplication.instance()
    if not isinstance(app, QApplication):
        app = QApplication(sys.argv[:1])
    _APP = app
    return app


@pytest.fixture(autouse=True)
def _qapp() -> None:
    _ensure_qapp()


def test_export_for_engine_yields_differing_engine_specific_content(
    tmp_path: Path,
) -> None:
    widget = MainWidget()
    try:
        # Load a model into the widget
        urdf_file = tmp_path / "simple.urdf"
        urdf_file.write_text(SIMPLE_URDF, encoding="utf-8")
        assert widget.load_model(urdf_file, notify=False)

        # Export for all three engines
        mujoco_out = widget.export_for_engine("MuJoCo")
        drake_out = widget.export_for_engine("Drake")
        pinocchio_out = widget.export_for_engine("Pinocchio")

        # 1. Output must not be identical across the three engines
        assert mujoco_out != drake_out, (
            "MuJoCo and Drake exports should not be identical"
        )
        assert drake_out != pinocchio_out, (
            "Drake and Pinocchio exports should not be identical"
        )
        assert mujoco_out != pinocchio_out, (
            "MuJoCo and Pinocchio exports should not be identical"
        )

        # 2. MuJoCo output must contain MuJoCo-specific MJCF/URDF structures
        assert "<mujoco" in mujoco_out, "MuJoCo export must include <mujoco> element"

        # 3. Drake output must contain Drake-specific tags/namespaces
        assert (
            "xmlns:drake" in drake_out or "drake:" in drake_out or "Drake" in drake_out
        ), "Drake export must contain Drake-specific content"
        assert "xmlns:drake" in drake_out or "drake:" in drake_out, (
            "Drake export must include Drake namespace or tags"
        )

        # 4. Pinocchio output must differ from Drake and contain Pinocchio identifier
        assert "Pinocchio" in pinocchio_out or "pinocchio" in pinocchio_out

        # Also verify writing to disk produces differing files
        f_mujoco = tmp_path / "out_mujoco.xml"
        f_drake = tmp_path / "out_drake.urdf"
        f_pin = tmp_path / "out_pin.urdf"

        widget.export_for_engine("MuJoCo", file_path=f_mujoco)
        widget.export_for_engine("Drake", file_path=f_drake)
        widget.export_for_engine("Pinocchio", file_path=f_pin)

        assert f_mujoco.read_text(encoding="utf-8") == mujoco_out
        assert f_drake.read_text(encoding="utf-8") == drake_out
        assert f_pin.read_text(encoding="utf-8") == pinocchio_out
    finally:
        widget.deleteLater()


def test_export_for_unsupported_engine_raises() -> None:
    widget = MainWidget()
    try:
        with pytest.raises(ValueError, match="Unsupported engine"):
            widget.export_for_engine("gazebo")

        with pytest.raises(ValueError, match="Unsupported engine"):
            widget.export_for_engine("jaxsim")

        with pytest.raises(ValueError, match="Unsupported engine"):
            widget.export_for_engine("invalid_engine")
    finally:
        widget.deleteLater()


def test_segment_manager_export_for_engine_content() -> None:
    mgr = SegmentManager()
    with pytest.raises((ValueError, AssertionError)):
        mgr.export_for_engine("gazebo")

    mujoco_res = mgr.export_for_engine("MuJoCo", urdf_content=SIMPLE_URDF)
    drake_res = mgr.export_for_engine("Drake", urdf_content=SIMPLE_URDF)
    pin_res = mgr.export_for_engine("Pinocchio", urdf_content=SIMPLE_URDF)

    assert mujoco_res["engine"] == "mujoco"
    assert drake_res["engine"] == "drake"
    assert pin_res["engine"] == "pinocchio"

    assert "<mujoco" in mujoco_res["content"]
    assert "drake" in drake_res["content"].lower()
    assert "pinocchio" in pin_res["content"].lower()
    assert mujoco_res["content"] != drake_res["content"]


def test_export_menu_excludes_unsupported_registered_engines() -> None:
    widget = MainWidget()
    try:
        file_menu = None
        menu_bar = widget.inner_main_window.menuBar()
        assert menu_bar is not None
        for action in menu_bar.actions():
            if action.text() == "&File":
                file_menu = action.menu()
                break

        assert file_menu is not None, "File menu not found"

        export_menu = None
        for action in file_menu.actions():
            if action.menu() and "Export" in action.text():
                export_menu = action.menu()
                break

        assert export_menu is not None, "Export menu not found"

        action_texts = [a.text() for a in export_menu.actions()]
        assert "Export for MuJoCo" in action_texts
        assert "Export for Drake" in action_texts
        assert "Export for Pinocchio" in action_texts

        # Other registered engines must be explicitly absent from the menu
        for unhonoured in ["OpenSim", "MyoSuite", "JaxSim", "Simscape"]:
            assert f"Export for {unhonoured}" not in action_texts
    finally:
        widget.deleteLater()


def test_viewer_button_precondition_and_empty_feedback() -> None:
    viewer = MuJoCoViewerWidget()
    try:
        # Precondition: disabled until model is loaded
        assert not viewer._launch_btn.isEnabled()

        # Calling _launch_external_viewer directly with empty content gives visible feedback
        with patch.object(QMessageBox, "warning") as mock_warning:
            viewer._launch_external_viewer()
            mock_warning.assert_called_once()
            assert "No URDF content" in viewer._status_label.text()

        # Loading model enables button (if MuJoCo is available)
        viewer.update_visualization(SIMPLE_URDF)
        if MUJOCO_AVAILABLE:
            assert viewer._launch_btn.isEnabled()

        # Clearing disables button again
        viewer.clear()
        assert not viewer._launch_btn.isEnabled()
    finally:
        viewer.deleteLater()


def test_viewer_button_error_feedback_on_converter_and_popen_failures() -> None:
    viewer = MuJoCoViewerWidget()
    try:
        viewer.update_visualization(SIMPLE_URDF)

        # 1. Converter failure (ValueError) is caught and reported
        with (
            patch(
                "src.tools.model_explorer.mujoco_viewer.URDFToMJCFConverter.convert",
                side_effect=ValueError("Invalid URDF geometry element"),
            ),
            patch.object(QMessageBox, "critical") as mock_critical,
        ):
            viewer._launch_external_viewer()
            mock_critical.assert_called_once()
            assert (
                "Failed to launch viewer" in viewer._status_label.text()
                or "Invalid URDF geometry" in viewer._status_label.text()
            )

        # 2. Subprocess failure (OSError) is caught and reported
        with (
            patch(
                "src.tools.model_explorer.mujoco_viewer.secure_popen",
                side_effect=OSError("Exec format error"),
            ),
            patch.object(QMessageBox, "critical") as mock_critical,
        ):
            viewer._launch_external_viewer()
            mock_critical.assert_called_once()
            assert (
                "Failed to launch viewer" in viewer._status_label.text()
                or "Exec format error" in viewer._status_label.text()
            )

        # 3. ImportError is caught and reported
        with (
            patch(
                "src.tools.model_explorer.mujoco_viewer.secure_popen",
                side_effect=ImportError("No module named mujoco"),
            ),
            patch.object(QMessageBox, "critical") as mock_critical,
        ):
            viewer._launch_external_viewer()
            mock_critical.assert_called_once()
            assert "Failed to launch viewer" in viewer._status_label.text()
    finally:
        viewer.deleteLater()
