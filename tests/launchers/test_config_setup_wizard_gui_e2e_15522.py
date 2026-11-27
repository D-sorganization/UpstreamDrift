"""Headless QTest E2E for the canonical-core Setup Wizard widget (#15522).

Launches the real ``ConfigSetupWizardWidget`` offscreen, types setup JSON into
its input with real key events and clicks the real Back / Validate / Next
buttons. Nothing is mocked: the deterministic ``SetupWizardViewModel`` runs
underneath, so these cases pin what a user would actually see.

The session-scoped ``qapp`` from ``tests/launchers/conftest.py`` owns the
``QApplication`` for the whole run; it is never constructed or discarded here.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

pytestmark = pytest.mark.unit

VALID_CONFIG: dict[str, Any] = {
    "convention": "canonical-v2",
    "units": "SI",
    "frame": "world_Zup",
    "model": {
        "canonical_id": "golfer-demo",
        "joint_names": ["hip", "spine"],
        "nq": 8,
        "nv": 7,
    },
    "calibration": {"status": "complete", "subject_id": "subject-001"},
}
UNITS_ONLY_CONFIG: dict[str, Any] = {
    "convention": "canonical-v2",
    "units": "SI",
    "frame": "world_Zup",
}


@pytest.fixture
def wizard(qapp):
    """Show a fresh wizard widget; skip when the tool tree cannot be imported."""
    try:
        from PyQt6 import QtCore, QtWidgets, sip

        from src.tools.config_setup_wizard.gui import ConfigSetupWizardWidget
    except (ImportError, OSError) as exc:  # e.g. vendor/ud-tools not checked out
        pytest.skip(f"setup wizard GUI not importable (vendor/ud-tools?): {exc}")

    widget = ConfigSetupWizardWidget()
    widget.resize(640, 480)
    widget.show()
    qapp.processEvents()
    assert widget.isVisible()
    buttons = {b.text(): b for b in widget.findChildren(QtWidgets.QPushButton)}
    yield widget, buttons
    if not sip.isdeleted(widget):
        widget.close()
        widget.deleteLater()
    qapp.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete.value)


def _click(button) -> None:
    from PyQt6 import QtCore, QtTest

    qtest: Any = QtTest.QTest  # stubs model the static helpers as methods
    qtest.mouseClick(button, QtCore.Qt.MouseButton.LeftButton)


def _type_json(widget, payload: str) -> None:
    """Replace the input text with ``payload`` using real key events."""
    from PyQt6 import QtTest

    widget._input.clear()
    widget._input.setFocus()
    qtest: Any = QtTest.QTest
    qtest.keyClicks(widget._input, payload)


def _status(widget) -> str:
    return widget._status.text()


def _output(widget) -> str:
    return widget._output.toPlainText()


def test_launch_shows_blocked_first_step_with_plain_language_fixes(wizard) -> None:
    widget, buttons = wizard
    assert set(buttons) == {"Back", "Validate", "Next"}
    assert _status(widget) == "Step: units frames | needs fixes"
    out = _output(widget)
    assert out.splitlines()[:4] == [
        "Units and Frames: blocked (3)",
        "Canonical Model: blocked (1)",
        "Calibration: blocked (1)",
        "Review: blocked (5)",
    ]
    assert 'Fix: Set "convention" to "canonical-v2".' in out
    assert widget._input.placeholderText() == "Paste canonical-core setup JSON"
    assert widget._output.isReadOnly()


def test_next_on_empty_input_and_back_at_first_step_do_not_move(wizard) -> None:
    widget, buttons = wizard
    _click(buttons["Next"])
    assert _status(widget).startswith("Step: units frames")
    _click(buttons["Back"])
    assert _status(widget).startswith("Step: units frames")


def test_validate_partial_config_unblocks_only_the_fixed_step(wizard) -> None:
    widget, buttons = wizard
    _type_json(widget, json.dumps(UNITS_ONLY_CONFIG))
    _click(buttons["Validate"])
    out = _output(widget)
    assert _status(widget) == "Step: units frames | needs fixes"
    assert "Units and Frames: ready (0)" in out
    assert "Canonical Model: blocked" in out
    assert "model: A canonical model block is required" in out
    assert "calibration: Subject calibration is required" in out


def test_full_walkthrough_next_to_review_then_back_to_start(wizard) -> None:
    widget, buttons = wizard
    _type_json(widget, json.dumps(VALID_CONFIG))
    expected = ["model", "calibration", "review"]
    for step in expected:
        _click(buttons["Next"])
        assert _status(widget) == f"Step: {step} | valid"
    _click(buttons["Next"])  # already on the last step: stays put
    assert _status(widget) == "Step: review | valid"
    out = _output(widget)
    assert "Units and Frames: complete (0)" in out
    assert "Review: ready (0)" in out
    for step in ("calibration", "model", "units frames"):
        _click(buttons["Back"])
        assert _status(widget) == f"Step: {step} | valid"


def test_invalid_json_and_non_object_report_input_errors_then_recover(
    wizard,
) -> None:
    widget, buttons = wizard
    _type_json(widget, "{not json")
    _click(buttons["Next"])
    assert _status(widget).startswith("Invalid input:")
    _type_json(widget, "[1, 2]")
    _click(buttons["Validate"])
    assert _status(widget) == "Invalid input: setup JSON must decode to an object"
    _type_json(widget, json.dumps(VALID_CONFIG))
    _click(buttons["Next"])
    assert _status(widget) == "Step: model | valid"


def test_malformed_gravity_is_reported_not_raised(wizard) -> None:
    widget, buttons = wizard
    bad = {**VALID_CONFIG, "gravity": [0.0, 0.0]}
    _type_json(widget, json.dumps(bad))
    _click(buttons["Next"])
    assert _status(widget) == "Step: units frames | needs fixes"
    assert "gravity: Gravity must be a finite 3-vector" in _output(widget)


def test_cleanup_schedules_deletion_of_the_widget(wizard, qapp) -> None:
    from PyQt6 import QtCore, sip

    widget, _ = wizard
    widget.cleanup()
    qapp.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete.value)
    assert sip.isdeleted(widget)
