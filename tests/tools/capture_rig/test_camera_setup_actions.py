"""Camera setup is reachable from the real header and optional wizard path."""

from pathlib import Path

import pytest
from PyQt6.QtTest import QTest

from src.motion_capture.rig.plan import CaptureMode
from src.tools.capture_rig.camera_setup import bind_camera, create_plan, save_revision
from src.tools.capture_rig.gui import CaptureRigWidget
from src.tools.capture_rig.record_bar import Phase
from tests.tools.capture_rig.test_camera_setup import camera
from tests.tools.capture_rig.test_pane_layout import _app, _settings

pytestmark = [pytest.mark.unit, pytest.mark.ui]


def test_selected_plan_survives_restart_and_clears_prior_overrides(
    tmp_path: Path,
) -> None:
    _app()
    settings = _settings(tmp_path)
    plan = create_plan("Lesson", [bind_camera("face_on", camera("one"), CaptureMode())])
    path = save_revision(plan, tmp_path / "library")
    widget = CaptureRigWidget(settings=settings)
    original_session = widget.capture.session_dir()
    try:
        widget.capture.mode_combo.setCurrentIndex(2)
        widget.capture.views_edit.setText("old_view")
        widget.capture.exposure_edit.setText("7")
        widget.camera_setup_actions.apply(path)
        assert widget.capture.selection().plan == path
        assert widget.capture.selection().mode is None
        assert not widget.capture.selection().views
        assert widget.capture.controls().exposure is None
        assert widget.capture.session_dir() == original_session
    finally:
        widget.shutdown()
    restored = CaptureRigWidget(settings=settings)
    try:
        assert restored.capture.selection().plan == path
        assert restored.camera_setup_actions.button.text() == "Camera Setup"
    finally:
        restored.shutdown()


def test_wizard_offers_camera_setup_without_making_it_an_import_prerequisite(
    tmp_path, monkeypatch
) -> None:
    _app()
    widget = CaptureRigWidget(settings=_settings(tmp_path))
    visits = []
    monkeypatch.setattr(
        widget.camera_setup_actions, "show", lambda: visits.append("setup")
    )
    try:
        widget.wizard_actions.show()
        wizard = widget.wizard_actions.dialog
        assert wizard is not None
        wizard.choose(["edit"])
        wizard.next()
        for _ in range(100):
            _app().processEvents()
            widget.wizard_actions._poll()
            if widget.wizard_actions._future is None:
                break
            QTest.qWait(10)
        page = wizard.step_pages["capture.library"]
        assert page.camera_setup_button is not None
        page.camera_setup_button.click()
        assert visits == ["setup"]
        assert wizard.route is not None and "step.setup" not in wizard.route.step_ids
        assert wizard.isVisible()
        wizard.close()
    finally:
        widget.shutdown()


def test_busy_capture_cannot_change_selected_plan(tmp_path: Path) -> None:
    _app()
    widget = CaptureRigWidget(settings=_settings(tmp_path))
    path = save_revision(
        create_plan("Lesson", [bind_camera("face_on", camera("one"), CaptureMode())]),
        tmp_path,
    )
    before = widget.capture.plan_edit.text()
    try:
        widget.record_bar.clock.phase = Phase.COUNTDOWN
        with pytest.raises(ValueError, match="Finish"):
            widget.camera_setup_actions.apply(path)
        assert widget.capture.plan_edit.text() == before
    finally:
        widget.record_bar.clock.phase = Phase.IDLE
        widget.shutdown()
