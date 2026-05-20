"""Tests for the runtime-mode help button (issue #4627)."""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")


@pytest.fixture(scope="module")
def qt_app():
    try:
        from PyQt6.QtWidgets import QApplication
    except (ImportError, OSError) as e:  # noqa: F841
        pytest.skip(f"PyQt6 runtime unavailable (import/runtime error): {e}")

    app = QApplication.instance() or QApplication(sys.argv[:1])
    yield app


def test_runtime_mode_help_button_uses_icon_not_text(qt_app) -> None:
    from src.launchers.runtime_mode_help import make_runtime_mode_help_button

    btn = make_runtime_mode_help_button()

    assert btn.text() == "", "runtime-mode help button must not use a literal '?'"
    assert not btn.icon().isNull(), "runtime-mode help button must carry an icon"


def test_runtime_mode_help_button_tooltip_is_specific(qt_app) -> None:
    from src.launchers.runtime_mode_help import make_runtime_mode_help_button

    btn = make_runtime_mode_help_button()
    tooltip = btn.toolTip()

    assert tooltip, "tooltip must be non-empty"
    # Tooltip should specifically mention the runtimes it explains, not be
    # a generic phrase like "More information".
    lower = tooltip.lower()
    assert "runtime" in lower or "native" in lower or "docker" in lower


def test_runtime_mode_help_button_accessible_name(qt_app) -> None:
    from src.launchers.runtime_mode_help import make_runtime_mode_help_button

    btn = make_runtime_mode_help_button()
    assert "runtime" in btn.accessibleName().lower()


def test_runtime_mode_help_button_click_opens_dialog(qt_app) -> None:
    from src.launchers.runtime_mode_help import make_runtime_mode_help_button

    with patch("src.launchers.runtime_mode_help.QMessageBox") as mock_msgbox_cls:
        instance = mock_msgbox_cls.return_value
        btn = make_runtime_mode_help_button()
        btn.click()

        assert mock_msgbox_cls.called, (
            "clicking the button must construct a QMessageBox"
        )
        # Find the title set on the message box.
        title_calls = [
            call_args.args[0] for call_args in instance.setWindowTitle.call_args_list
        ]
        assert any("Engine Runtime" in t for t in title_calls), (
            f"expected 'Engine Runtime' in dialog title; got {title_calls!r}"
        )
