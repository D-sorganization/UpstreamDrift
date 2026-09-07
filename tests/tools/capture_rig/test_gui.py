"""Capture Rig widget: guided workflow, command building, playback, results."""

from __future__ import annotations

import json
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
from src.tools.capture_rig.workflow import Status
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
    widget.process.separate_set_check.setChecked(True)
    ing = widget.command_for("ingest")
    assert ing[ing.index("--estimator") + 1] == "openpose_dnn"
    assert ing[ing.index("--max-frames") + 1] == "50"
    assert "--option" in ing and "input_height=368" in ing and "min_peak=0.1" in ing
    assert ing[-1].endswith("observations_openpose_dnn")
    widget.process.estimator_combo.setCurrentIndex(
        widget.process.estimator_combo.findData("mediapipe")
    )
    assert widget.process.options()["model_variant"] == "full"
    assert widget.process.options()["enable_temporal_smoothing"] is True
    widget.process.start_edit.setText(str(tmp_path / "intrinsics.json"))
    widget.process.exclude_edit.setText("nose, left_ankle")
    rc = widget.command_for("reconstruct")
    assert "--intrinsics" in rc and rc[rc.index("--anchor") + 1] == "neck=0.53"
    assert rc[rc.index("--exclude-joints") + 1] == "nose,left_ankle"
    widget.process.start_edit.setText(str(tmp_path / "reconstruction.json"))
    assert "--cameras" in widget.command_for("reconstruct")
    assert widget.command_for("calibrate")[-4:] == [
        "--square",
        "0.025",
        "--every",
        "10",
    ]
    widget.capture.pending_import = [("face_on", tmp_path / "a.mp4")]
    imp = widget.command_for("import")
    assert imp[3] == "import" and imp[-1] == f"face_on={tmp_path / 'a.mp4'}"
    assert widget.command_for("compare")[3] == "compare"
    assert widget.command_for("reliability")[3] == "reliability"
    assert widget.command_for("analyze")[3] == "analyze"
    assert widget.command_for("export")[3] == "export"
    with pytest.raises(ValueError):
        widget.command_for("nope")


def test_without_a_session_only_setup_actions_are_enabled() -> None:
    _app()
    widget = CaptureRigWidget()
    assert widget.enabled_actions() == {"plan_check", "import", "stop", "load"}
    assert widget.workflow.statuses()["setup"] is Status.READY
    widget.capture.plan_edit.setText("")
    widget.trigger("plan_check")
    assert "choose a plan file" in widget.log.toPlainText()
    assert not widget.runner.busy


def test_session_load_drives_workflow_playback_and_results(tmp_path: Path) -> None:
    _app()
    root = _bundle(tmp_path)
    _observations(root)
    (root / "reconstruct").mkdir()
    (root / "reconstruct" / "swing_summary.json").write_text(
        '{"x_factor_deg": 41.0}', encoding="utf-8"
    )
    (root / "reliability.json").write_text(
        json.dumps(
            {
                "joints": [
                    {"joint": "left_hip", "score": 0.9},
                    {"joint": "nose", "score": 0.2},
                ],
                "recommended_exclusions": ["nose"],
            }
        ),
        encoding="utf-8",
    )
    (root / "intrinsics.json").write_text("[]", encoding="utf-8")
    widget = CaptureRigWidget()
    widget.capture.session_edit.setText(str(root))
    media = widget.refresh_session()
    assert media is not None and media.ingested
    statuses = widget.workflow.statuses()
    assert statuses["capture"] is Status.DONE and statuses["detect"] is Status.DONE
    assert statuses["intrinsics"] is Status.DONE  # two planned views, file present
    assert statuses["export"] is Status.BLOCKED  # no session_reconstruction yet
    enabled = widget.enabled_actions()
    assert {"ingest", "compare", "reliability", "reconstruct"} <= enabled
    assert "analyze" not in enabled  # multi-camera session
    assert widget.process.exclude_edit.text() == "nose"
    assert widget.process.start_edit.text() == str(root / "intrinsics.json")
    assert widget.playback.view_combo.count() == 1  # cam_b has no recording
    assert widget.playback.set_combo.count() == 1
    assert widget.playback.image.pixmap() is not None
    widget.playback.show_frame(3)
    assert widget.playback.status.text().endswith("· pose")
    widget.playback.show_frame(2)
    assert widget.playback.status.text().endswith("no pose")
    widget.playback.toggle_play()
    widget.playback.show_frame(99)
    widget.playback.step()
    assert not widget.playback.playing
    assert widget.swing_table.rowCount() == 1
    assert widget.reliability_table.item(0, 0).text() == "left_hip"
    assert "reliable" in widget.reliability_table.item(0, 1).text()
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
