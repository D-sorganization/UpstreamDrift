"""Tests for launcher-owned Sidekick tab dispatch glue."""

from __future__ import annotations

from typing import Any
from src.launchers.launcher_dialogs import DialogsManager


class _LauncherHarness:
    def __init__(self, sidebar=None, embedded_host=None) -> None:
        self.sidekick_sidebar = sidebar
        self.embedded_host = embedded_host
        self.toasts: list[tuple[str, str]] = []
        self.dialogs_manager = DialogsManager(self)

    def show_toast(self, message: str, toast_type: str = "info") -> None:
        self.toasts.append((message, toast_type))

    def __getattr__(self, name: str) -> Any:
        if name == "dialogs_manager":
            raise AttributeError
        if hasattr(self.dialogs_manager, name):
            attr = getattr(self.dialogs_manager, name)
            import types

            if isinstance(attr, types.MethodType):
                return types.MethodType(attr.__func__, self)
            return attr
        raise AttributeError(name)


class _Sidebar:
    def __init__(self) -> None:
        self.visible = False
        self.visible_tabs = {"sidekick"}
        self.active_tab = "sidekick"

    def setVisible(self, value: bool) -> None:  # noqa: N802 - Qt-style API
        self.visible = value

    def set_active_tab(self, tab_id: str) -> bool:
        if tab_id not in self.visible_tabs:
            return False
        self.active_tab = tab_id
        return True

    def set_tab_visible(self, tab_id: str, visible: bool) -> bool:
        if visible:
            self.visible_tabs.add(tab_id)
        else:
            self.visible_tabs.discard(tab_id)
        return True


class _EmbeddedHost:
    def __init__(self) -> None:
        self.opened: list[str] = []

    def open_tab(self, tool_id: str) -> None:
        self.opened.append(tool_id)


def test_open_sidekick_tab_activates_visible_tools_sidebar_tab() -> None:
    sidebar = _Sidebar()
    sidebar.visible_tabs.add("os_terminal")
    launcher = _LauncherHarness(sidebar=sidebar)

    launcher.open_sidekick_tab("os_terminal")

    assert sidebar.visible is True
    assert sidebar.active_tab == "os_terminal"
    assert launcher.toasts == []


def test_open_sidekick_tab_unhides_then_activates_tools_sidebar_tab() -> None:
    sidebar = _Sidebar()
    launcher = _LauncherHarness(sidebar=sidebar)

    launcher.open_sidekick_tab("python_repl")

    assert "python_repl" in sidebar.visible_tabs
    assert sidebar.active_tab == "python_repl"
    assert launcher.toasts == []


def test_open_sidekick_tab_falls_back_to_embedded_host() -> None:
    host = _EmbeddedHost()
    launcher = _LauncherHarness(sidebar=None, embedded_host=host)

    launcher.open_sidekick_tab("workspace")

    assert host.opened == ["workspace"]
    assert launcher.toasts == []
