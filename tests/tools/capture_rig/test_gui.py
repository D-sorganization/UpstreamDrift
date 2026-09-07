"""Capture Rig widget: panels build the right commands and playback draws frames."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")
pytest.importorskip("cv2")
from PyQt6.QtWidgets import QApplication

from src.tools.capture_rig import commands
from src.tools.capture_rig.gui import CaptureRigWidget, get_dockable_ui
from tests.tools.capture_rig.test_core import _bundle, _observations

pytestmark = [pytest.mark.unit, pytest.mark.ui]


def _app() -> QApplication:
    app = QApplication.instance()
    return app if app is not None else QApplication(sys.argv[:1])


def test_panels_build_commands_from_their_inputs(tmp_path: Path) -> None:
    _app()
    widget = CaptureRigWidget()
    widget.capture.plan_edit.setText(str(tmp_path / "plan.json"))
    widget.capture.session_edit.setText(str(tmp_path / "take"))
    widget.capture.mode_combo.setCurrentIndex(1)  # first preset
    widget.capture.views_edit.setText("cam_b, cam_c")
    widget.capture.exposure_edit.setText("-7")
    widget.capture.auto_exposure_combo.setCurrentText("off")
    widget.capture.duration_spin.setValue(30)
    widget.capture.dry_run_check.setChecked(True)
    rec = widget.command_for("record")
    assert rec[3] == "record" and rec[-1] == "--dry-run"
    assert rec[rec.index("--mode") + 1] == commands.mode_text(commands.MODE_PRESETS[0])
    assert rec[rec.index("--views") + 1] == "cam_b,cam_c"
    assert rec[rec.index("--exposure") + 1] == "-7"
    assert rec[rec.index("--auto-exposure") + 1] == "off"
    assert rec[rec.index("--duration") + 1] == "30"
    widget.process.estimator_combo.setCurrentIndex(
        widget.process.estimator_combo.findData("openpose_dnn")
    )
    widget.process.max_frames_spin.setValue(50)
    ing = widget.command_for("ingest")
    assert ing[-4:] == ["--estimator", "openpose_dnn", "--max-frames", "50"]
    widget.process.start_edit.setText(str(tmp_path / "intrinsics.json"))
    rc = widget.command_for("reconstruct")
    assert "--intrinsics" in rc and rc[rc.index("--anchor") + 1] == "neck=0.53"
    widget.process.start_edit.setText(str(tmp_path / "reconstruction.json"))
    assert "--cameras" in widget.command_for("reconstruct")
    assert widget.command_for("calibrate")[-4:] == [
        "--square",
        "0.025",
        "--every",
        "10",
    ]
    with pytest.raises(ValueError):
        widget.command_for("nope")


def test_missing_inputs_are_logged_not_raised(tmp_path: Path) -> None:
    _app()
    widget = CaptureRigWidget()
    widget.capture.plan_edit.setText("")
    widget.trigger("plan_check")
    assert "choose a plan file" in widget.log.toPlainText()
    assert not widget.runner.busy


def test_session_load_drives_playback_and_results(tmp_path: Path) -> None:
    _app()
    root = _bundle(tmp_path)
    _observations(root)
    (root / "reconstruct").mkdir()
    (root / "reconstruct" / "swing_summary.json").write_text(
        '{"x_factor_deg": 41.0}', encoding="utf-8"
    )
    widget = CaptureRigWidget()
    widget.capture.session_edit.setText(str(root))
    media = widget.refresh_session()
    assert media is not None and media.ingested
    assert widget.playback.view_combo.count() == 1  # cam_b has no recording
    assert widget.playback.image.pixmap() is not None
    widget.playback.show_frame(3)
    assert widget.playback.frame_index == 3
    assert widget.playback.status.text().endswith("· pose")
    widget.playback.show_frame(2)
    assert widget.playback.status.text().endswith("no pose")
    widget.playback.show_frame(99)
    assert widget.playback.frame_index == 11
    widget.playback.toggle_play()
    assert widget.playback.playing
    widget.playback.step()
    assert not widget.playback.playing  # stepping past the end stops playback
    assert widget.results.rowCount() == 1
    assert widget.results.item(0, 0).text() == "x_factor_deg"
    widget.playback.close_media()


def test_dockable_ui_and_adapter() -> None:
    _app()
    window = get_dockable_ui()
    assert window.windowTitle() == "Capture Rig"
    from src.tools.capture_rig._embed_adapter import CaptureRigAdapter

    adapter = CaptureRigAdapter()
    assert adapter.tool_id == "capture_rig" and not adapter.is_dirty()
    child = adapter.create_main_widget(None)
    assert isinstance(child, CaptureRigWidget)
    adapter.cleanup()
    assert not adapter.is_dirty()
