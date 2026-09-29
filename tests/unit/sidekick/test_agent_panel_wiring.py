"""AIAssistantPanel wiring for launcher-owned Sidekick actions (#7209)."""

from __future__ import annotations

import os
import sys
from typing import Any

import pytest

from src.launchers.sidekick_host_port import create_launcher_action_service

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

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
    def open_sidekick_tab(self, tool_id: str) -> None:
        self.opened = tool_id


@pytest.fixture
def qapp() -> Any:
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication(sys.argv[:1])


def _panel(qapp: Any, tmp_path, monkeypatch: pytest.MonkeyPatch) -> Any:
    del qapp
    import src.shared.python.ai.gui.assistant_panel as panel_mod

    monkeypatch.setattr(panel_mod.AIAssistantPanel, "_load_history", lambda self: None)
    return panel_mod.AIAssistantPanel(project_root=tmp_path)


def test_panel_without_action_service_exposes_no_sidekick_tools(
    qapp, tmp_path, monkeypatch
) -> None:
    panel = _panel(qapp, tmp_path, monkeypatch)

    assert panel._sidekick_tool_declarations() == []
    assert "sidekick_system_prompt" not in panel._context.metadata


def test_panel_set_action_service_adds_prompt_and_tool_catalog(
    qapp, tmp_path, monkeypatch
) -> None:
    panel = _panel(qapp, tmp_path, monkeypatch)
    service = create_launcher_action_service(
        launcher=_Launcher(),
        embedded_host=_Host(),
    )

    panel.set_action_service(service)
    tools = panel._sidekick_tool_declarations()

    assert service._dispatcher is panel._main_thread_dispatcher
    assert "subtab.list" in panel._context.metadata["sidekick_system_prompt"]
    assert any(
        tool.get("function", {}).get("name") == "sidekick.action.subtab.list"
        for tool in tools
    )


def test_panel_invokes_sidekick_actions_on_gui_thread(
    qapp, tmp_path, monkeypatch
) -> None:
    host = _Host()
    panel = _panel(qapp, tmp_path, monkeypatch)
    panel.set_action_service(
        create_launcher_action_service(
            launcher=_Launcher(),
            embedded_host=host,
        )
    )

    result = panel.invoke_sidekick_action(
        "subtab.focus",
        {"tab_id": "workspace"},
    )

    assert result.ok is True
    assert host.focused == ["workspace"]
