"""Headless PyQt6 end-to-end tests for the MyoSuite dashboard GUI (#15522).

The MyoSuite engine is an optional wheel plus an optional git submodule, so
the dashboard probes lazily. These tests drive the *real* ``MainWindow`` /
``MainWidget`` the way a user does -- real ``QTest`` mouse clicks on
*Probe engine*, key presses into the read-only usage pane, and a real window
close -- and control only the environment (wheel flag, submodule directory
and the lazily imported engine class), never the Qt widgets.
"""

from __future__ import annotations

import os
import sys
import types
from collections.abc import Iterator
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication, QPushButton, QTextEdit  # noqa: E402

from src.engines.physics_engines.myosuite.python import gui as myo_gui  # noqa: E402
from src.shared.python.engine_core import engine_availability  # noqa: E402

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
_ENGINE_MODULE = "src.engines.physics_engines.myosuite.python.myosuite_physics_engine"


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    return app  # type: ignore[return-value]


@pytest.fixture
def window(qapp: QApplication) -> Iterator[myo_gui.MainWindow]:
    win = myo_gui.MainWindow()
    win.show()
    QTest.qWaitForWindowExposed(win)
    yield win
    win.close()
    win.deleteLater()


def _set_wheel(monkeypatch: pytest.MonkeyPatch, available: bool) -> None:
    monkeypatch.setattr(
        engine_availability, "MYOSUITE_AVAILABLE", available, raising=False
    )


def _fake_submodule(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Make any ``.../myo_sim`` directory look populated; return queried paths."""
    queried: list[Path] = []
    real_exists, real_iterdir = Path.exists, Path.iterdir

    def exists(self: Path, *args: object, **kwargs: object) -> bool:
        if self.name == "myo_sim":
            queried.append(self)
            return True
        return real_exists(self, *args, **kwargs)  # type: ignore[arg-type]

    def iterdir(self: Path) -> Iterator[Path]:
        if self.name == "myo_sim":
            return iter([self / "model.xml"])
        return real_iterdir(self)

    monkeypatch.setattr(Path, "exists", exists)
    monkeypatch.setattr(Path, "iterdir", iterdir)
    return queried


def _install_engine(monkeypatch: pytest.MonkeyPatch, engine_cls: type) -> None:
    module = types.ModuleType(_ENGINE_MODULE)
    module.MyoSuitePhysicsEngine = engine_cls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, _ENGINE_MODULE, module)


def _click_probe(win: myo_gui.MainWindow) -> None:
    button = win.main_widget.findChild(QPushButton)
    assert button is not None and button.text() == "Probe engine"
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)


def _status(win: myo_gui.MainWindow) -> str:
    return win.main_widget._status_label.text()


def test_window_launches_with_dashboard_content(window: myo_gui.MainWindow) -> None:
    assert window.isVisible()
    assert window.windowTitle() == "MyoSuite Dashboard"
    assert window.centralWidget() is window.main_widget
    assert isinstance(window.main_widget, myo_gui.MainWidget)
    # The probe runs once at construction so the user never sees "not probed".
    assert _status(window).startswith("Engine status:")
    assert "not probed" not in _status(window)
    usage = window.main_widget.findChild(QTextEdit)
    assert usage is not None and "Engine()" in usage.toPlainText()


def test_probe_click_without_wheel_explains_how_to_enable(
    window: myo_gui.MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_wheel(monkeypatch, False)
    window.main_widget._status_label.setText("stale")

    _click_probe(window)

    text = _status(window)
    assert "unavailable" in text
    assert "pip install myosuite" in text


def test_probe_click_with_wheel_and_submodule_reports_available(
    window: myo_gui.MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_wheel(monkeypatch, True)
    _fake_submodule(monkeypatch)
    instances: list[object] = []

    class FakeEngine:
        def __init__(self) -> None:
            instances.append(self)

    _install_engine(monkeypatch, FakeEngine)
    window.main_widget._status_label.setText("stale")

    _click_probe(window)

    assert _status(window) == "Engine status: available"
    assert len(instances) == 1


def test_probe_click_surfaces_engine_construction_failure(
    window: myo_gui.MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_wheel(monkeypatch, True)
    _fake_submodule(monkeypatch)

    class BrokenEngine:
        def __init__(self) -> None:
            raise RuntimeError("mujoco exploded")

    _install_engine(monkeypatch, BrokenEngine)

    _click_probe(window)

    text = _status(window)
    assert "unavailable (mujoco exploded)" in text
    assert "pip install myosuite" in text


def test_probe_is_repeatable_and_tracks_environment_changes(
    window: myo_gui.MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_wheel(monkeypatch, False)
    _click_probe(window)
    assert "unavailable" in _status(window)

    _set_wheel(monkeypatch, True)
    _fake_submodule(monkeypatch)
    _install_engine(monkeypatch, type("Ok", (), {}))
    _click_probe(window)
    assert _status(window) == "Engine status: available"


def test_submodule_check_looks_inside_this_repository(
    window: myo_gui.MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ``myo_sim`` submodule lives under the repo root, not above it."""
    _set_wheel(monkeypatch, True)
    queried = _fake_submodule(monkeypatch)
    _install_engine(monkeypatch, type("Ok", (), {}))

    _click_probe(window)

    assert queried, "probe never checked for the myo_sim submodule"
    expected = REPO_ROOT / "shared" / "models" / "myosuite" / "myo_sim"
    assert queried[-1] == expected


def test_usage_snippet_is_read_only_under_keyboard_input(
    window: myo_gui.MainWindow,
) -> None:
    usage = window.main_widget.findChild(QTextEdit)
    assert usage is not None and usage.isReadOnly()
    before = usage.toPlainText()

    usage.setFocus()
    QTest.keyClicks(usage, "tampered")
    QTest.keyClick(usage, Qt.Key.Key_Backspace)

    assert usage.toPlainText() == before


def test_closing_window_runs_widget_cleanup_once(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    win = myo_gui.MainWindow()
    win.show()
    calls: list[int] = []
    monkeypatch.setattr(win.main_widget, "cleanup", lambda: calls.append(1))

    assert win.close()

    assert calls == [1]
    assert not win.isVisible()


def test_dockable_entry_point_and_embedded_widget_cleanup(
    qapp: QApplication,
) -> None:
    docked = myo_gui.get_dockable_ui()
    assert isinstance(docked, myo_gui.MainWindow)
    docked.close()

    embedded = myo_gui.MainWidget()
    embedded.cleanup()
    embedded.cleanup()  # idempotent: the host may call it more than once
    embedded.close()
