"""Shared ``python -m`` launcher for single-widget desktop tools (#14177)."""

from __future__ import annotations

import os
import sys
from typing import Any

import pytest

from src.tools import standalone_window

pytestmark = pytest.mark.unit


class _FakeApp:
    """Stands in for ``QApplication`` so the event loop never blocks."""

    def exec(self) -> int:
        return 0


class _FakeWidget:
    instances: list[_FakeWidget] = []

    def __init__(self) -> None:
        self.size: tuple[int, int] | None = None
        self.shown = False
        _FakeWidget.instances.append(self)

    def resize(self, width: int, height: int) -> None:
        self.size = (width, height)

    def show(self) -> None:
        self.shown = True


def test_rejects_a_widget_path_without_a_class_name() -> None:
    with pytest.raises(ValueError, match="module:ClassName"):
        standalone_window.run_standalone_widget(
            "src.tools.putting_analytics_gui.gui", (1, 1)
        )


def test_rejects_a_non_positive_size() -> None:
    with pytest.raises(ValueError, match="positive"):
        standalone_window.run_standalone_widget("pkg.mod:Widget", (0, 10))


def test_missing_gui_dependency_prints_the_install_hint(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = standalone_window.run_standalone_widget(
        "src.tools.no_such_tool_for_test.gui:MainWidget", (800, 600)
    )
    assert code == 1
    err = capsys.readouterr().err
    assert "pip install -e '.[gui-tools]'" in err
    assert "no_such_tool_for_test" in err


def test_opens_the_widget_at_the_requested_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    module: Any = type(sys)("fake_tool_gui_for_test")
    module.MainWidget = _FakeWidget
    monkeypatch.setitem(sys.modules, "fake_tool_gui_for_test", module)
    monkeypatch.setattr(standalone_window, "_application", lambda: _FakeApp())

    code = standalone_window.run_standalone_widget(
        "fake_tool_gui_for_test:MainWidget", (1100, 760)
    )

    assert code == 0
    widget = _FakeWidget.instances[-1]
    assert widget.size == (1100, 760) and widget.shown
