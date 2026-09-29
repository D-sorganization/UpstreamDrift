"""Tests for the pendulum equations popup facade (Issue #2388)."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

MODULE_NAME = "src.shared.python.pendulum_simulator.gui.equations_popup"
MODULE_PREFIX = f"{MODULE_NAME}_"


class _Signal:
    def __init__(self) -> None:
        self._callback = None

    def connect(self, callback) -> None:
        self._callback = callback

    def emit(self) -> None:
        if self._callback is not None:
            self._callback()


class _FakeClipboard:
    def __init__(self) -> None:
        self.text = ""

    def setText(self, text: str) -> None:
        self.text = text


class _FakeDialog:
    def __init__(self, parent=None) -> None:
        self.parent = parent
        self.widgets: list[object] = []
        self.window_title = ""
        self.minimum_size = (0, 0)
        self.stylesheet = ""
        self.attribute = None
        self.shown = False

    def setWindowTitle(self, title: str) -> None:
        self.window_title = title

    def setMinimumSize(self, width: int, height: int) -> None:
        self.minimum_size = (width, height)

    def setStyleSheet(self, stylesheet: str) -> None:
        self.stylesheet = stylesheet

    def setAttribute(self, attribute) -> None:
        self.attribute = attribute

    def show(self) -> None:
        self.shown = True


class _FakeVBoxLayout:
    def __init__(self, dialog: _FakeDialog) -> None:
        self.dialog = dialog
        self.contents_margins: tuple[()] | tuple[int, int, int, int] = ()

    def setContentsMargins(self, left: int, top: int, right: int, bottom: int) -> None:
        self.contents_margins = (left, top, right, bottom)

    def addWidget(self, widget: object) -> None:
        self.dialog.widgets.append(widget)


class _FakeTextBrowser:
    def __init__(self) -> None:
        self.open_external_links = False
        self.html = ""
        self.stylesheet = ""

    def setOpenExternalLinks(self, enabled: bool) -> None:
        self.open_external_links = enabled

    def setHtml(self, html: str) -> None:
        self.html = html

    def setStyleSheet(self, stylesheet: str) -> None:
        self.stylesheet = stylesheet

    def toPlainText(self) -> str:
        return "Rendered equation text"


class _FakePushButton:
    def __init__(self, text: str) -> None:
        self.text = text
        self.clicked = _Signal()

    def click(self) -> None:
        self.clicked.emit()


def _install_fake_pyqt(monkeypatch: pytest.MonkeyPatch) -> _FakeClipboard:
    clipboard = _FakeClipboard()
    pyqt6: Any = ModuleType("PyQt6")
    qtcore: Any = ModuleType("PyQt6.QtCore")
    qtwidgets: Any = ModuleType("PyQt6.QtWidgets")

    class _FakeApplication:
        @staticmethod
        def clipboard() -> _FakeClipboard:
            return clipboard

    qtcore.Qt = SimpleNamespace(
        WidgetAttribute=SimpleNamespace(WA_DeleteOnClose="delete-on-close")
    )
    qtwidgets.QApplication = _FakeApplication
    qtwidgets.QDialog = _FakeDialog
    qtwidgets.QPushButton = _FakePushButton
    qtwidgets.QTextBrowser = _FakeTextBrowser
    qtwidgets.QVBoxLayout = _FakeVBoxLayout
    pyqt6.QtCore = qtcore
    pyqt6.QtWidgets = qtwidgets

    monkeypatch.setitem(sys.modules, "PyQt6", pyqt6)
    monkeypatch.setitem(sys.modules, "PyQt6.QtCore", qtcore)
    monkeypatch.setitem(sys.modules, "PyQt6.QtWidgets", qtwidgets)
    return clipboard


def _reload_equations_popup():
    for name in list(sys.modules):
        if name == MODULE_NAME or name.startswith(MODULE_PREFIX):
            sys.modules.pop(name, None)
    return importlib.import_module(MODULE_NAME)


def test_show_equations_popup_supports_all_topics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clipboard = _install_fake_pyqt(monkeypatch)
    equations_popup = _reload_equations_popup()

    for topic in equations_popup.EquationTopic:
        dialog = equations_popup.show_equations_popup(None, topic)

        assert dialog.window_title
        assert dialog.minimum_size == (720, 600)
        assert dialog.attribute == "delete-on-close"
        assert dialog.shown is True
        assert len(dialog.widgets) == 2

        browser, copy_button = dialog.widgets
        assert browser.open_external_links is True
        assert browser.html.lstrip().startswith("<html><head><style>")
        assert copy_button.text == "Copy to Clipboard"

        clipboard.text = ""
        copy_button.click()
        assert clipboard.text == "Rendered equation text"


def test_show_equations_popup_rejects_unknown_topic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_pyqt(monkeypatch)
    equations_popup = _reload_equations_popup()

    with pytest.raises(ValueError, match="Unknown topic"):
        equations_popup.show_equations_popup(None, "unknown")
