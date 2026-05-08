"""Unit tests for the shared info-icon button helper."""

from __future__ import annotations

import os
import sys

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


def test_info_button_has_icon_and_no_question_mark_text(qt_app) -> None:
    from PyQt6.QtCore import QSize

    from src.shared.python.ui.info_button import make_info_button

    btn = make_info_button(tooltip="Specific help text", accessible_name="Help A")

    assert not btn.icon().isNull(), "info button should carry a non-null icon"
    assert btn.iconSize() == QSize(16, 16)
    assert btn.text() == "", "info button must NOT use a literal '?' text glyph"


def test_info_button_invokes_callback(qt_app) -> None:
    from src.shared.python.ui.info_button import make_info_button

    calls: list[int] = []

    def _cb() -> None:
        calls.append(1)

    btn = make_info_button(on_click=_cb, tooltip="X", accessible_name="X")
    btn.click()

    assert calls == [1]


def test_info_button_accessible_name(qt_app) -> None:
    from src.shared.python.ui.info_button import make_info_button

    btn = make_info_button(
        tooltip="Adjust the size of the model tiles",
        accessible_name="Tile scale help",
    )

    assert btn.accessibleName() == "Tile scale help"
    assert btn.toolTip() == "Adjust the size of the model tiles"


def test_info_button_custom_icon_size(qt_app) -> None:
    from PyQt6.QtCore import QSize

    from src.shared.python.ui.info_button import make_info_button

    btn = make_info_button(icon_size_px=24, tooltip="X", accessible_name="X")
    assert btn.iconSize() == QSize(24, 24)


def test_info_button_is_auto_raise(qt_app) -> None:
    from src.shared.python.ui.info_button import make_info_button

    btn = make_info_button(tooltip="X", accessible_name="X")
    assert btn.autoRaise() is True
