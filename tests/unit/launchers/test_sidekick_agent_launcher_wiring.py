"""Focused launcher wiring tests for Sidekick agent mode (#7209)."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from src.launchers.sidekick_host_port import create_launcher_action_service

pytestmark = pytest.mark.unit


class _Host:
    def __init__(self) -> None:
        self.opened = ["workspace", "model_explorer"]
        self.focused: list[str] = []

    def open_tool_ids(self) -> list[str]:
        return list(self.opened)

    def active_tool_id(self) -> str | None:
        return self.opened[-1] if self.opened else None

    def focus_tab(self, tool_id: str) -> None:
        if tool_id not in self.opened:
            raise KeyError(tool_id)
        self.focused.append(tool_id)

    def open_tab(self, tool_id: str) -> int:
        if tool_id not in self.opened:
            self.opened.append(tool_id)
        return self.opened.index(tool_id)

    def close_tab(self, target: int | str, *, destroy: bool = True) -> bool:
        del destroy
        if target not in self.opened:
            return False
        self.opened.remove(str(target))
        return True

    def backgrounded_tools(self) -> set[str]:
        return set()

    def popped_out_tools(self) -> set[str]:
        return set()


class _Launcher:
    def __init__(self) -> None:
        self.opened: list[str] = []
        self.orchestrator = types.SimpleNamespace(
            available_models={"model_explorer": object(), "workspace": object()}
        )

    def open_sidekick_tab(self, tool_id: str) -> None:
        self.opened.append(tool_id)


def test_launcher_action_service_registers_subtab_and_host_actions() -> None:
    host = _Host()
    launcher = _Launcher()

    service = create_launcher_action_service(launcher=launcher, embedded_host=host)
    action_ids = {descriptor.action_id for descriptor in service.list_actions()}

    assert "subtab.list" in action_ids
    assert "subtab.focus" in action_ids
    assert "host.launcher.list_tiles" in action_ids
    assert "host.launcher.open_tile" in action_ids

    focused = service.invoke("subtab.focus", {"tab_id": "model_explorer"})
    opened = service.invoke("host.launcher.open_tile", {"tool_id": "workspace"})

    assert focused.ok is True
    assert host.focused == ["model_explorer"]
    assert opened.ok is True
    assert launcher.opened == ["workspace"]


def test_sidekick_embed_adapter_injects_parent_action_service(monkeypatch) -> None:
    from src.tools.sidekick._embed_adapter import _SidekickEmbedAdapter

    service = create_launcher_action_service(
        launcher=_Launcher(),
        embedded_host=_Host(),
    )
    seen: list[Any] = []

    class _Panel:
        def __init__(self, parent: Any) -> None:
            self.parent = parent

        def set_action_service(self, injected: Any) -> None:
            seen.append(injected)

    module = types.ModuleType("src.shared.python.ai.gui.assistant_panel")
    module.AIAssistantPanel = _Panel
    monkeypatch.setitem(sys.modules, module.__name__, module)

    parent = types.SimpleNamespace(sidekick_action_service=service)
    widget = _SidekickEmbedAdapter().create_main_widget(parent)

    assert isinstance(widget, _Panel)
    assert seen == [service]
