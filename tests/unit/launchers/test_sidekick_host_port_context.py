"""LauncherSubtabPort integration with LauncherContext workspace (#7210)."""

from __future__ import annotations

import pytest

from src.launchers.sidekick_host_port import LauncherSubtabPort
from src.shared.python.launcher_embed.context import InMemoryLauncherContext

pytestmark = pytest.mark.unit


class _Host:
    def __init__(self) -> None:
        self.opened: list[str] = []

    def open_tool_ids(self) -> list[str]:
        return list(self.opened)

    def active_tool_id(self) -> str | None:
        return self.opened[-1] if self.opened else None

    def focus_tab(self, tool_id: str) -> None:
        if tool_id not in self.opened:
            raise KeyError(tool_id)

    def open_tab(self, tool_id: str) -> int:
        self.opened.append(tool_id)
        return len(self.opened) - 1

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


def test_launcher_subtab_port_can_use_launcher_context_workspace() -> None:
    context = InMemoryLauncherContext()
    port = LauncherSubtabPort(_Host(), workspace=context)
    seen: list[dict[str, object]] = []
    context.subscribe("value_changed:club", seen.append)

    prior = port.workspace_set_variable("club", "driver")

    assert prior is None
    assert port.workspace_snapshot().values == {"club": "driver"}
    assert context.get_value("club") == "driver"
    assert seen[-1]["value"] == "driver"
