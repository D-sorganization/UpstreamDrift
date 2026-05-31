"""Headless unit tests for :class:`EmbeddedHostWidget` backgrounding.

Covers the Sub-PR A capabilities of issue #6013:

- background-close stashes the widget and calls ``pause()``;
- reopen re-surfaces the stashed widget and calls ``resume()``;
- destroy-close calls ``cleanup()``;
- ``pop_out_tab`` / ``dock_back`` re-parent the live widget;
- ``backgrounded_tools()`` reflects the stashed set;
- a tool with ``can_background() is False`` gets legacy cleanup;
- a tool with ``detach_to_window() is False`` is pin-only.

All tests run under ``QT_QPA_PLATFORM=offscreen`` and skip cleanly when
PyQt6 is not installed.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtWidgets import (  # noqa: E402
    QApplication,
    QLabel,
    QMainWindow,
    QWidget,
)

from src.launchers.embedded_host import EmbeddedHostWidget  # noqa: E402
from src.shared.python.launcher_embed import (  # noqa: E402
    EmbedCapabilities,
    register_embeddable_tool,
    unregister_embeddable_tool,
)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = [pytest.mark.unit]


class _LifecycleTool:
    """Adapter that records every optional-hook invocation.

    ``can_background`` / ``detach_to_window`` are configurable so a
    single fixture covers the backgroundable, legacy-cleanup, and
    pin-only cases.
    """

    def __init__(
        self,
        tool_id: str,
        *,
        can_bg: bool = True,
        detachable: bool = True,
    ) -> None:
        self.tool_id = tool_id
        self._can_bg = can_bg
        self._detachable = detachable
        self.pause_calls = 0
        self.resume_calls = 0
        self.cleanup_calls = 0
        self.create_calls = 0

    def embed_capabilities(self) -> EmbedCapabilities:
        return EmbedCapabilities()

    def create_main_widget(self, parent: Any) -> QWidget:
        self.create_calls += 1
        label = QLabel("content", parent)
        label.setObjectName(f"lifecycle::{self.tool_id}")
        return label

    def cleanup(self) -> None:
        self.cleanup_calls += 1

    def is_dirty(self) -> bool:
        return False

    def pause(self) -> None:
        self.pause_calls += 1

    def resume(self) -> None:
        self.resume_calls += 1

    def can_background(self) -> bool:
        return self._can_bg

    def detach_to_window(self) -> bool:
        return self._detachable


class _LegacyTool:
    """Adapter that omits the optional #6013 hooks entirely.

    Verifies the structural ``getattr``-with-default behaviour for the
    ~17 pre-existing adapters that have not been updated.
    """

    def __init__(self, tool_id: str) -> None:
        self.tool_id = tool_id
        self.cleanup_calls = 0

    def embed_capabilities(self) -> EmbedCapabilities:
        return EmbedCapabilities()

    def create_main_widget(self, parent: Any) -> QWidget:
        return QLabel("legacy", parent)

    def cleanup(self) -> None:
        self.cleanup_calls += 1

    def is_dirty(self) -> bool:
        return False


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def host(qapp):  # noqa: ANN001
    widget = EmbeddedHostWidget()
    yield widget
    widget.close()
    widget.deleteLater()


def _register(tool: Any) -> None:
    """Register ``tool`` and schedule its removal after the test."""
    register_embeddable_tool(tool)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Snapshot registered ids so each test leaves the registry clean."""
    registered: list[str] = []
    yield registered
    for tool_id in registered:
        unregister_embeddable_tool(tool_id)


# ----------------------------------------------------------------------
# Background-close
# ----------------------------------------------------------------------


def test_background_close_stashes_and_pauses(host, _clean_registry) -> None:  # noqa: ANN001
    tool = _LifecycleTool("bg_tool")
    _register(tool)
    _clean_registry.append(tool.tool_id)

    host.open_tab(tool.tool_id)
    assert host.close_tab(tool.tool_id, destroy=False) is True

    assert tool.pause_calls == 1
    assert tool.cleanup_calls == 0
    assert host.backgrounded_tools() == {"bg_tool"}
    assert "bg_tool" not in host.active_tool_ids()


def test_reopen_resurfaces_stashed_widget_and_resumes(host, _clean_registry) -> None:  # noqa: ANN001
    tool = _LifecycleTool("bg_tool")
    _register(tool)
    _clean_registry.append(tool.tool_id)

    host.open_tab(tool.tool_id)
    host.close_tab(tool.tool_id, destroy=False)

    host.open_tab(tool.tool_id)

    # The widget was re-surfaced, not rebuilt.
    assert tool.create_calls == 1
    assert tool.resume_calls == 1
    assert host.backgrounded_tools() == set()
    assert "bg_tool" in host.active_tool_ids()


def test_destroy_close_calls_cleanup(host, _clean_registry) -> None:  # noqa: ANN001
    tool = _LifecycleTool("destroy_tool")
    _register(tool)
    _clean_registry.append(tool.tool_id)

    host.open_tab(tool.tool_id)
    assert host.close_tab(tool.tool_id, destroy=True) is True

    assert tool.cleanup_calls == 1
    assert tool.pause_calls == 0
    assert host.backgrounded_tools() == set()
    assert "destroy_tool" not in host.active_tool_ids()


def test_cannot_background_tool_gets_legacy_cleanup(host, _clean_registry) -> None:  # noqa: ANN001
    tool = _LifecycleTool("no_bg_tool", can_bg=False)
    _register(tool)
    _clean_registry.append(tool.tool_id)

    host.open_tab(tool.tool_id)
    # Even requesting a background close falls back to destroy because
    # the tool cannot be backgrounded.
    assert host.close_tab(tool.tool_id, destroy=False) is True

    assert tool.cleanup_calls == 1
    assert tool.pause_calls == 0
    assert host.backgrounded_tools() == set()


def test_legacy_tool_without_hooks_backgrounds(host, _clean_registry) -> None:  # noqa: ANN001
    tool = _LegacyTool("legacy_tool")
    _register(tool)
    _clean_registry.append(tool.tool_id)

    host.open_tab(tool.tool_id)
    # No pause()/can_background() defined -> structural default True,
    # so a non-destroy close stashes without error.
    assert host.close_tab(tool.tool_id, destroy=False) is True
    assert host.backgrounded_tools() == {"legacy_tool"}
    assert tool.cleanup_calls == 0


# ----------------------------------------------------------------------
# Pop-out / dock-back
# ----------------------------------------------------------------------


def test_pop_out_reparents_into_top_level_window(host, _clean_registry) -> None:  # noqa: ANN001
    tool = _LifecycleTool("pop_tool")
    _register(tool)
    _clean_registry.append(tool.tool_id)

    host.open_tab(tool.tool_id)
    assert host.pop_out_tab(tool.tool_id) is True

    assert "pop_tool" in host.active_tool_ids()
    assert "pop_tool" not in host.backgrounded_tools()
    # Tab is gone from the central tab widget.
    assert host.tab_widget.count() == 0
    # The popped-out widget now lives under a top-level QMainWindow.
    popped = host._popped_out["pop_tool"]
    assert isinstance(popped.window, QMainWindow)
    assert popped.window.centralWidget() is popped.widget
    # No fresh widget was constructed.
    assert tool.create_calls == 1


def test_dock_back_returns_widget_to_tab(host, _clean_registry) -> None:  # noqa: ANN001
    tool = _LifecycleTool("pop_tool")
    _register(tool)
    _clean_registry.append(tool.tool_id)

    host.open_tab(tool.tool_id)
    host.pop_out_tab(tool.tool_id)

    index = host.dock_back(tool.tool_id)

    assert index == 0
    assert host.tab_widget.count() == 1
    assert "pop_tool" not in host._popped_out
    assert host.tab_widget.indexOf(tool_widget := host.tab_widget.widget(0)) == 0
    assert tool_widget is not None
    assert tool.create_calls == 1


def test_reopen_popped_out_tool_docks_it_back(host, _clean_registry) -> None:  # noqa: ANN001
    tool = _LifecycleTool("pop_tool")
    _register(tool)
    _clean_registry.append(tool.tool_id)

    host.open_tab(tool.tool_id)
    host.pop_out_tab(tool.tool_id)

    # Reopening a popped-out tool re-docks rather than rebuilding.
    host.open_tab(tool.tool_id)
    assert host.tab_widget.count() == 1
    assert "pop_tool" not in host._popped_out
    assert tool.create_calls == 1


def test_pin_only_tool_refuses_pop_out(host, _clean_registry) -> None:  # noqa: ANN001
    tool = _LifecycleTool("pin_tool", detachable=False)
    _register(tool)
    _clean_registry.append(tool.tool_id)

    host.open_tab(tool.tool_id)
    assert host.pop_out_tab(tool.tool_id) is False

    # Still mounted as a tab; nothing was popped out.
    assert host.tab_widget.count() == 1
    assert host._popped_out == {}


def test_pop_out_unknown_tool_returns_false(host) -> None:  # noqa: ANN001
    assert host.pop_out_tab("not_open") is False


def test_dock_back_unknown_tool_returns_minus_one(host) -> None:  # noqa: ANN001
    assert host.dock_back("not_popped") == -1


# ----------------------------------------------------------------------
# backgrounded_tools() set semantics
# ----------------------------------------------------------------------


def test_backgrounded_tools_tracks_multiple(host, _clean_registry) -> None:  # noqa: ANN001
    tool_a = _LifecycleTool("a")
    tool_b = _LifecycleTool("b")
    for tool in (tool_a, tool_b):
        _register(tool)
        _clean_registry.append(tool.tool_id)
        host.open_tab(tool.tool_id)

    host.close_tab("a", destroy=False)
    host.close_tab("b", destroy=False)
    assert host.backgrounded_tools() == {"a", "b"}

    host.open_tab("a")
    assert host.backgrounded_tools() == {"b"}
