"""Standard wizard navigation exposes requirements without launching hidden jobs."""

import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QWizard

from src.tools.capture_rig.goal_catalog import load_catalog
from src.tools.capture_rig.goal_planner import Readiness
from src.tools.capture_rig.goal_wizard import CaptureWizard
from tests.tools.capture_rig.test_pane_layout import _app

pytestmark = [pytest.mark.unit, pytest.mark.ui]


def test_edit_route_has_working_actions_and_gates_next_until_inputs_exist() -> None:
    _app()
    wizard = CaptureWizard(load_catalog())
    actions = []
    wizard.action_requested.connect(actions.append)
    wizard.choose(["edit"])
    wizard.show()
    wizard.next()
    assert wizard.route is not None
    assert wizard.route.step_ids == ("capture.library", "capture.selection")
    wizard.update_evidence("No Capture", {"capture.library": Readiness("ready")})
    assert not wizard.button(QWizard.WizardButton.NextButton).isEnabled()
    wizard.step_pages["capture.library"].open_button.click()
    assert actions == ["capture.library"]
    wizard.update_evidence(
        "Swing One",
        {
            "capture.library": Readiness("done"),
            "capture.selection": Readiness("ready", "Save the selected swing"),
        },
    )
    wizard.next()
    assert wizard.current_step == "capture.selection"
    assert not wizard.button(QWizard.WizardButton.FinishButton).isEnabled()
    wizard.back()
    assert wizard.current_step == "capture.library"
    wizard.close()


def test_optional_club_step_can_be_skipped_but_required_steps_cannot() -> None:
    _app()
    wizard = CaptureWizard(load_catalog())
    wizard.choose(["fit_model"])
    wizard.show()
    wizard.next()
    assert wizard.step_pages["clubs.bag"].skip is not None
    assert wizard.step_pages["step.detect"].skip is None
    requests = []
    wizard.skip_requested.connect(lambda key, skipped: requests.append((key, skipped)))
    wizard.step_pages["clubs.bag"].skip.setChecked(True)
    assert requests == [("clubs.bag", True)]
    wizard.close()


def test_incompatible_goals_stay_on_choice_page_with_explanation() -> None:
    _app()
    wizard = CaptureWizard(load_catalog())
    wizard.choose(["analyze_2d", "reconstruct"])
    wizard.show()
    wizard.next()
    assert wizard.currentId() == 0
    assert "incompatible" in wizard.choices.feedback.text().lower()
    wizard.close()


def test_saved_route_can_reopen_at_its_step_and_busy_state_blocks_actions() -> None:
    _app()
    wizard = CaptureWizard(load_catalog())
    wizard.choose(["edit"])
    wizard.show()
    wizard.next()
    wizard.navigate("capture.selection")
    assert wizard.current_step == "capture.selection"
    wizard.update_evidence("Swing One", {}, busy=True)
    assert not wizard.step_pages["capture.selection"].open_button.isEnabled()
    assert "running" in wizard.step_pages["capture.selection"].status.text().lower()
    wizard.close()


def test_small_window_keeps_navigation_visible() -> None:
    app = _app()
    wizard = CaptureWizard(load_catalog())
    wizard.resize(660, 560)
    wizard.choose(["edit"])
    wizard.show()
    for _ in range(3):
        app.processEvents()
    for role in (
        QWizard.WizardButton.NextButton,
        QWizard.WizardButton.CancelButton,
        QWizard.WizardButton.CustomButton1,
    ):
        button = wizard.button(role)
        assert button.isVisible()
        bottom = button.mapTo(wizard, button.rect().bottomRight())
        assert 0 < bottom.y() <= wizard.height()
        assert 0 < bottom.x() <= wizard.width()
    wizard.next()
    wizard.update_evidence("Swing One", {"capture.library": Readiness("done")})
    wizard.next()
    app.processEvents()
    assert wizard.button(QWizard.WizardButton.BackButton).isVisible()
    wizard.close()
