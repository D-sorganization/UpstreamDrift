"""Native camera setup paths use synthetic discovery and real plan persistence."""

import subprocess
from pathlib import Path
from threading import Event

import pytest
from PyQt6.QtTest import QTest

from src.motion_capture.rig.plan import CameraControls, CaptureMode
from src.tools.capture_rig.camera_setup import bind_camera, create_plan, load_plan
from src.tools.capture_rig.camera_setup_dialog import CameraSetupDialog
from src.tools.capture_rig.commands import PlanSelection, plan_check_command
from tests.tools.capture_rig.test_camera_setup import camera
from tests.tools.capture_rig.test_pane_layout import _app

pytestmark = [pytest.mark.unit, pytest.mark.ui]


def finish_scan(dialog: CameraSetupDialog) -> None:
    for _ in range(100):
        _app().processEvents()
        dialog.poll_scan()
        if dialog._future is None:
            return
        QTest.qWait(10)
    pytest.fail("Synthetic discovery did not finish")


def test_save_reopen_and_real_plan_command(tmp_path: Path) -> None:
    _app()
    dialog = CameraSetupDialog(tmp_path)
    devices = [camera("one"), camera("two")]
    devices[1].index = 1
    dialog.apply_devices(devices)
    dialog.rows[0].view.setText("face_on")
    dialog.rows[1].view.setText("down_line")
    dialog.rows[0].mode.setCurrentText("1280x720@120:MJPG")
    dialog.notes.setPlainText("Face-on camera beside the ball")
    dialog.check_connections()
    assert "All views match" in dialog.status.text()
    dialog.save()
    assert dialog.saved_path is not None
    plan = load_plan(dialog.saved_path)
    assert plan.cameras[0].view == "face_on" and plan.cameras[0].mode.fps == 120
    assert str(dialog.saved_path) in plan_check_command(
        PlanSelection(dialog.saved_path)
    )
    reopened = CameraSetupDialog(tmp_path, plan=plan)
    try:
        assert reopened.plan() == plan
        reopened.apply_devices([])
        reopened.check_connections()
        assert "face_on" in reopened.status.text()
        assert reopened.plan() == plan
    finally:
        reopened.reject()


def test_duplicate_choice_cannot_write_a_plan(tmp_path: Path) -> None:
    _app()
    dialog = CameraSetupDialog(tmp_path / "library")
    try:
        dialog.apply_devices([camera("one"), camera("two")])
        dialog.rows[1].device.setCurrentIndex(1)
        dialog.save()
        assert dialog.saved_path is None
        assert "duplicate" in dialog.status.text()
        assert not (tmp_path / "library").exists()
    finally:
        dialog.reject()


def test_scan_cancel_preserves_controls_and_allows_rescan(tmp_path: Path) -> None:
    _app()
    binding = bind_camera("face_on", camera("one"), CaptureMode()).model_copy(
        update={"controls": CameraControls(exposure=7)}
    )
    plan = create_plan("Saved", [binding])
    dialog = CameraSetupDialog(
        tmp_path,
        plan=plan,
        discover=lambda event: (event.wait(0.2), [camera("two")])[1],
    )
    try:
        dialog.scan()
        dialog.cancel_scan()
        finish_scan(dialog)
        assert dialog.plan() == plan and dialog.scan_button.isEnabled()
        assert "cancelled" in dialog.status.text().lower()
        dialog._discover = lambda _event: [camera("one")]
        dialog.scan()
        finish_scan(dialog)
        assert dialog.plan() == plan
    finally:
        dialog.reject()


def test_discovery_failure_offers_recovery(tmp_path: Path) -> None:
    _app()

    def unavailable(_event):
        raise OSError("Camera access unavailable")

    dialog = CameraSetupDialog(tmp_path, discover=unavailable)
    try:
        dialog.scan()
        finish_scan(dialog)
        assert "Camera access unavailable" in dialog.status.text()
        assert "import video" in dialog.status.text()
        assert dialog.scan_button.isEnabled()
    finally:
        dialog.reject()


def test_discovery_timeout_preserves_setup_and_hides_probe_command(
    tmp_path: Path,
) -> None:
    _app()
    plan = create_plan("Saved", [bind_camera("face_on", camera("one"), CaptureMode())])

    def timeout(_event: Event) -> list:
        raise subprocess.TimeoutExpired("internal-probe-command", 12)

    dialog = CameraSetupDialog(tmp_path, plan=plan, discover=timeout)
    try:
        dialog.scan()
        finish_scan(dialog)
        assert dialog.plan() == plan
        assert "timed out" in dialog.status.text()
        assert "scan again" in dialog.status.text()
        assert "internal-probe-command" not in dialog.status.text()
        assert dialog.scan_button.isEnabled()
    finally:
        dialog.reject()
