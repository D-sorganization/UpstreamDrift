from __future__ import annotations

from unittest.mock import Mock

import pytest


def _import_qt():
    try:
        from PyQt6.QtWidgets import QWidget
    except (ImportError, OSError) as exc:
        pytest.skip(f"PyQt6 not loadable: {exc}")

    from src.shared.python.gui_pkg.gui_utils import get_qapp

    get_qapp()
    return QWidget


def test_custom_title_bar_can_hide_secondary_close_button() -> None:
    _import_qt()

    from src.launchers.custom_title_bar import CustomTitleBar

    title_bar = CustomTitleBar(show_close_button=False)

    assert title_bar.btn_close is None


class _DummySignal:
    def __init__(self) -> None:
        self._slot = None

    def connect(self, slot) -> None:
        self._slot = slot

    def emit(self) -> None:
        assert self._slot is not None
        self._slot()


class _DummyButton:
    def __init__(self) -> None:
        self.clicked = _DummySignal()


def test_menu_bar_close_widget_uses_launcher_callback(monkeypatch) -> None:
    QWidget = _import_qt()

    from src.launchers import launcher_ui_setup

    callback = Mock()
    button = _DummyButton()
    monkeypatch.setattr(
        launcher_ui_setup,
        "create_window_control_button",
        lambda *args, **kwargs: button,
    )

    launcher_ui_setup._build_menu_bar_close_widget(QWidget(), callback)
    button.clicked.emit()

    callback.assert_called_once_with()
