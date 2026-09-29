"""Extra coverage for src.launchers.runtime_mode_help.

Complements ``tests/launchers/test_runtime_mode_help.py`` by exercising
the HTML payload structure and the ``show_runtime_mode_help`` dialog
factory, without relying on monkey-patched info-button internals.
"""

from __future__ import annotations

from unittest.mock import patch

from src.launchers import runtime_mode_help as rmh


def test_html_payload_mentions_all_three_runtimes() -> None:
    """All three documented runtimes must appear in the help text."""
    html = rmh.RUNTIME_MODE_HELP_HTML
    assert "Native Windows" in html
    assert "Docker" in html
    assert "WSL2" in html
    # Sanity: the HTML lists the engines that the launcher supports
    for engine in ("MuJoCo", "Drake", "Pinocchio", "OpenSim", "MyoSuite"):
        assert engine in html


def test_show_runtime_mode_help_constructs_and_executes_dialog(qapp) -> None:
    """The help dialog should be configured with the canonical HTML."""
    captured: dict[str, object] = {}

    # Snapshot real enums before patching
    real_icon = rmh.QMessageBox.Icon
    real_buttons = rmh.QMessageBox.StandardButton

    class _FakeBox:
        Icon = real_icon
        StandardButton = real_buttons

        def __init__(self, parent=None):
            captured["parent"] = parent

        def setWindowTitle(self, title):
            captured["title"] = title

        def setIcon(self, icon):
            captured["icon"] = icon

        def setTextFormat(self, fmt):
            captured["text_format"] = fmt

        def setText(self, text):
            captured["text"] = text

        def setStandardButtons(self, btns):
            captured["buttons"] = btns

        def exec(self):
            captured["executed"] = True

    with patch.object(rmh, "QMessageBox", _FakeBox):
        rmh.show_runtime_mode_help(parent=None)

    assert captured["executed"] is True
    assert captured["title"] == "Engine Runtime — what does this control?"
    assert captured["text"] == rmh.RUNTIME_MODE_HELP_HTML


def test_public_exports() -> None:
    assert set(rmh.__all__) == {
        "RUNTIME_MODE_HELP_HTML",
        "make_runtime_mode_help_button",
        "show_runtime_mode_help",
    }
