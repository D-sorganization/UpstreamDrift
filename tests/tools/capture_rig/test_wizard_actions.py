"""The header wizard uses existing screens and safely resumes the selected swing."""

import json
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QWizard

from src.motion_capture.rig.edits import SessionEdits, save_edits
from src.tools.capture_rig.gui import CaptureRigWidget
from src.tools.capture_rig.goal_planner import CaptureGoalRequest
from src.tools.capture_rig.library_actions import LIBRARY_ROOT_KEY
from tests.motion_capture.rig.test_ingest import _bundle
from tests.tools.capture_rig.test_pane_layout import _app, _settings

pytestmark = [pytest.mark.unit, pytest.mark.ui]


@pytest.fixture
def capture_host(tmp_path):
    _app()
    settings = _settings(tmp_path)
    settings.setValue(LIBRARY_ROOT_KEY, str(tmp_path / "library"))
    host = CaptureRigWidget(settings=settings)
    yield host
    actions = host.wizard_actions
    if actions.dialog is not None:
        actions.dialog.close()
    _ready(actions)
    host.close()
    host.deleteLater()
    _app().processEvents()


def _ready(actions) -> None:
    for _ in range(200):
        _app().processEvents()
        if actions._future is None:
            return
        QTest.qWait(10)
    raise AssertionError("Wizard status inspection did not complete")


def test_no_camera_library_navigation_and_cancel_are_visible(
    capture_host, monkeypatch
) -> None:
    actions = capture_host.wizard_actions
    actions.button.click()
    wizard = actions.dialog
    assert wizard is not None
    wizard.choose(["edit"])
    wizard.next()
    _ready(actions)
    assert "No Capture" in wizard.step_pages["capture.library"].identity.text()
    opened = []
    monkeypatch.setattr(
        capture_host.library_actions, "show_library", lambda: opened.append("library")
    )
    wizard.step_pages["capture.library"].open_button.click()
    _ready(actions)
    assert opened == ["library"]
    assert wizard.current_step == "capture.library"
    assert not capture_host.runner.busy
    wizard.reject()
    assert capture_host.media is None


def test_edit_save_resume_and_changed_inputs_require_review(
    capture_host, tmp_path
) -> None:
    root = _bundle(tmp_path)
    capture_host._open_library_capture(root)
    actions = capture_host.wizard_actions
    actions.show()
    wizard = actions.dialog
    wizard.choose(["edit"])
    wizard.next()
    _ready(actions)
    wizard.next()
    _ready(actions)
    assert wizard.current_step == "capture.selection"
    save_edits(root, SessionEdits())
    actions.refresh()
    _ready(actions)
    assert wizard.button(QWizard.WizardButton.FinishButton).isEnabled()
    actions.save()
    assert (root / "capture_workflow.json").is_file()
    actions.show()
    _ready(actions)
    actions.resume()
    _ready(actions)
    assert wizard.current_step == "capture.selection"
    (root / "session_manifest.json").write_text(
        json.dumps(
            json.loads((root / "session_manifest.json").read_text())
            | {"operator_note": "changed"}
        ),
        encoding="utf-8",
    )
    actions.resume()
    assert "needs review" in capture_host.journey.message.text().lower()
    assert "changed" in capture_host.journey.message.text().lower()


def test_map_plan_loads_validated_goal_and_unknown_plan_keeps_selection(
    capture_host, tmp_path, monkeypatch
) -> None:
    from src.tools.capture_rig import wizard_actions

    actions = capture_host.wizard_actions
    actions.show()
    wizard = actions.dialog
    plan = tmp_path / "capture-plan.json"
    request = CaptureGoalRequest(
        goals=("draw",), catalog_revision=wizard.catalog.revision
    )
    plan.write_text(request.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(
        wizard_actions.QFileDialog, "getOpenFileName", lambda *_args: (str(plan), "")
    )
    actions.open_plan()
    assert wizard.choices.checks["draw"].isChecked()
    plan.write_text(
        request.model_dump_json().replace('"draw"', '"unknown"'), encoding="utf-8"
    )
    actions.open_plan()
    assert wizard.choices.checks["draw"].isChecked()
    assert "unknown" in wizard.choices.feedback.text().lower()


def test_my_clubs_action_uses_the_existing_bag(capture_host, monkeypatch) -> None:
    actions = capture_host.wizard_actions
    actions.show()
    wizard = actions.dialog
    wizard.choose(["fit_model"])
    wizard.next()
    _ready(actions)
    opened = []
    monkeypatch.setattr(
        capture_host.equipment_actions, "show_equipment", lambda: opened.append("bag")
    )
    actions.open_step("clubs.bag")
    _ready(actions)
    assert opened == ["bag"]
    assert not capture_host.runner.busy


def test_failed_optional_runtime_restores_controls_and_explains_recovery(
    capture_host, tmp_path, monkeypatch
) -> None:
    root = _bundle(tmp_path)
    capture_host._open_library_capture(root)
    actions = capture_host.wizard_actions
    actions.show()
    wizard = actions.dialog
    wizard.choose(["reconstruct"])
    wizard.next()
    _ready(actions)
    monkeypatch.setattr(
        capture_host,
        "command_for",
        lambda _action: [str(tmp_path / "unavailable-detector.exe")],
    )
    capture_host.trigger("ingest")
    for _ in range(100):
        _app().processEvents()
        if capture_host.journey_actions.active_action is None:
            break
        QTest.qWait(10)
    _ready(actions)
    assert "failed" in capture_host.journey.message.text().lower()
    assert "retry" in capture_host.journey.message.text().lower()
    assert wizard.step_pages["step.detect"].open_button.isEnabled()
    assert not wizard.step_pages["step.detect"].isComplete()
    assert not capture_host.runner.busy


def test_unavailable_calibration_keeps_native_editing_and_recovery_links_usable(
    capture_host, tmp_path
) -> None:
    from tests.tools.capture_rig.test_wizard_evidence import _reviewed_calibration

    root = _bundle(tmp_path)
    capture_host._open_library_capture(root)
    actions = capture_host.wizard_actions
    actions.show()
    review, path = _reviewed_calibration(root, tmp_path)
    capture_host.process.start_edit.setText(str(path))
    actions.review = review
    wizard = actions.dialog
    path.unlink()
    wizard.choose(["reconstruct"])
    wizard.next()
    _ready(actions)
    calibration = wizard.step_pages["step.intrinsics"]
    assert calibration.open_button.isEnabled()
    assert "calibration" in calibration.status.text().lower()
    assert not calibration.isComplete()
    assert wizard.step_pages["capture.library"].isComplete()
    wizard.restart()
    wizard.choose(["edit"])
    wizard.next()
    _ready(actions)
    selection = wizard.step_pages["capture.selection"]
    assert selection.open_button.isEnabled()
    save_edits(root, SessionEdits())
    actions.refresh()
    _ready(actions)
    wizard.next()
    _ready(actions)
    assert selection.isComplete()
    assert wizard.button(QWizard.WizardButton.FinishButton).isEnabled()


def test_saved_edit_journey_resumes_in_a_fresh_capture_window(
    capture_host, tmp_path
) -> None:
    root = _bundle(tmp_path)
    capture_host._open_library_capture(root)
    actions = capture_host.wizard_actions
    actions.show()
    wizard = actions.dialog
    wizard.choose(["edit"])
    wizard.next()
    _ready(actions)
    wizard.next()
    _ready(actions)
    save_edits(root, SessionEdits())
    actions.refresh()
    _ready(actions)
    actions.save()
    saved = (root / "capture_workflow.json").read_bytes()
    edits = (root / "swing_edits.json").read_bytes()
    wizard.close()
    _ready(actions)
    capture_host.close()

    reopened = CaptureRigWidget(settings=_settings(tmp_path))
    resumed = reopened.wizard_actions
    try:
        reopened._open_library_capture(root)
        resumed.show()
        _ready(resumed)
        resumed.resume()
        _ready(resumed)
        assert resumed.dialog is not wizard
        assert resumed.dialog.current_step == "capture.selection"
        assert resumed.dialog.button(QWizard.WizardButton.FinishButton).isEnabled()
        assert resumed.review is None
        assert (root / "capture_workflow.json").read_bytes() == saved
        assert (root / "swing_edits.json").read_bytes() == edits
    finally:
        if resumed.dialog is not None:
            resumed.dialog.close()
        _ready(resumed)
        reopened.close()
        reopened.deleteLater()
        _app().processEvents()
