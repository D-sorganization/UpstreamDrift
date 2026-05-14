"""Tests for optional Unified Tools Sidebar host integration."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from src.shared.python.gui_launcher.tools_sidebar_integration import (
    install_tools_sidebar,
)


class FakeSignal:
    def __init__(self) -> None:
        self.slots: list[Any] = []

    def connect(self, slot: Any) -> None:
        self.slots.append(slot)

    def emit(self, value: Any) -> None:
        for slot in self.slots:
            slot(value)


class FakeDock:
    def __init__(self) -> None:
        self.file_open_requested = FakeSignal()
        self.widget_value = None

    def setWidget(self, widget: Any) -> None:
        self.widget_value = widget

    def toggleViewAction(self) -> object:
        return object()


class FakeMainWindow:
    def __init__(self) -> None:
        self.docks: list[tuple[Any, Any]] = []
        self.opened_paths: list[Any] = []

    def addDockWidget(self, area: Any, dock: Any) -> None:
        self.docks.append((area, dock))

    def open_file(self, path: Any) -> None:
        self.opened_paths.append(path)


def test_install_tools_sidebar_noops_when_shared_module_is_missing() -> None:
    status = install_tools_sidebar(FakeMainWindow())

    assert status.installed is False
    assert "not available" in status.reason


def test_install_tools_sidebar_adds_shared_dock_and_connects_file_open(
    monkeypatch: Any, tmp_path: Path
) -> None:
    module = ModuleType("upstream_drift_tools.ui.tools_sidebar")
    created: dict[str, Any] = {}

    class ToolsSidebar(FakeDock):
        def __init__(
            self,
            *,
            parent: Any,
            project_root: Path,
            context_provider: Any,
        ) -> None:
            super().__init__()
            created["parent"] = parent
            created["project_root"] = project_root
            created["context_provider"] = context_provider

    module.ToolsSidebar = ToolsSidebar  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)

    window = FakeMainWindow()
    status = install_tools_sidebar(
        window,
        project_root=tmp_path,
        context_provider=lambda: {"host": "test"},
    )

    assert status.installed is True
    assert status.file_open_connected is True
    assert status.module_name == module.__name__
    assert len(window.docks) == 1
    assert created["parent"] is window
    assert created["project_root"] == tmp_path.resolve()

    status.sidebar.file_open_requested.emit("model.urdf")

    assert window.opened_paths == ["model.urdf"]


def test_install_tools_sidebar_passes_sidekick_tokens_when_supported(
    monkeypatch: Any,
) -> None:
    module = ModuleType("upstream_drift_tools.ui.tools_sidebar")
    created: dict[str, Any] = {}

    class ToolsSidebar(FakeDock):
        def __init__(
            self,
            *,
            parent: Any,
            sidekick_tokens: dict[str, str],
        ) -> None:
            super().__init__()
            created["parent"] = parent
            created["sidekick_tokens"] = sidekick_tokens

    module.ToolsSidebar = ToolsSidebar  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)

    window = FakeMainWindow()
    status = install_tools_sidebar(window)

    assert status.installed is True
    assert created["parent"] is window
    assert created["sidekick_tokens"]["sidekick.color.surface"]
    assert created["sidekick_tokens"]["sidekick.radius.chat"] == "8px"


def test_install_tools_sidebar_uses_shared_installer_status(monkeypatch: Any) -> None:
    module = ModuleType("upstream_drift_tools.ui.tools_sidebar")
    dock = FakeDock()

    def shared_installer(
        main_window: Any,
        *,
        project_root: Path | None,
        context_provider: Any,
    ) -> FakeDock:
        main_window.addDockWidget("right", dock)
        return dock

    module.install_tools_sidebar = shared_installer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)

    window = FakeMainWindow()
    status = install_tools_sidebar(window)

    assert status.installed is True
    assert status.reason == "installed by shared module"
    assert status.dock is dock
    assert status.file_open_connected is True


def test_install_tools_sidebar_shared_installer_can_accept_sidekick_tokens(
    monkeypatch: Any,
) -> None:
    module = ModuleType("upstream_drift_tools.ui.tools_sidebar")
    observed: dict[str, Any] = {}
    dock = FakeDock()

    def shared_installer(
        main_window: Any,
        *,
        sidekick_tokens: dict[str, str],
    ) -> FakeDock:
        observed["tokens"] = sidekick_tokens
        main_window.addDockWidget("right", dock)
        return dock

    module.install_tools_sidebar = shared_installer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)

    status = install_tools_sidebar(FakeMainWindow())

    assert status.installed is True
    assert observed["tokens"]["sidekick.color.canvas"]


def test_install_tools_sidebar_rejects_non_dock_hosts(monkeypatch: Any) -> None:
    module = ModuleType("upstream_drift_tools.ui.tools_sidebar")
    module.ToolsSidebar = FakeDock  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)

    status = install_tools_sidebar(object())

    assert status.installed is False
    assert "dock widgets" in status.reason
