"""Pop-out / dock-back lifecycle hooks and tab introspection.

Covers the pause/resume bracket around tab re-parenting
(:meth:`EmbeddedHostWidget.pop_out_tab` / :meth:`dock_back`) and the
public tab-introspection API (``open_tool_ids``, ``active_tool_id``,
``focus_tab``) added for the Sidekick subtab port.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication, QLabel  # noqa: E402

from src.launchers.embedded_host import EmbeddedHostWidget  # noqa: E402
from src.shared.python.launcher_embed import (  # noqa: E402
    EmbedCapabilities,
    EMBEDDABLE_TOOL_REGISTRY,
    register_embeddable_tool,
)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = [pytest.mark.unit]


@pytest.fixture(scope="module")
def _qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class _HookTool:
    """EmbeddableTool that records pause/resume calls."""

    def __init__(self, tool_id: str) -> None:
        self.tool_id = tool_id
        self.pause_calls = 0
        self.resume_calls = 0

    def embed_capabilities(self) -> EmbedCapabilities:
        return EmbedCapabilities(supports_embedded=True)

    def create_main_widget(self, parent: Any) -> Any:
        return QLabel(self.tool_id, parent)

    def cleanup(self) -> None:
        pass

    def is_dirty(self) -> bool:
        return False

    def pause(self) -> None:
        self.pause_calls += 1

    def resume(self) -> None:
        self.resume_calls += 1


@pytest.fixture
def host(_qapp):  # noqa: ANN001
    w = EmbeddedHostWidget()
    yield w
    w.close()
    w.deleteLater()


@pytest.fixture
def tool(request):  # noqa: ANN001
    tool_id = f"hooktool::{request.node.name}"
    t = _HookTool(tool_id)
    register_embeddable_tool(t)
    yield t
    EMBEDDABLE_TOOL_REGISTRY.pop(tool_id, None)


class TestPopOutHooks:
    def test_pop_out_brackets_with_pause_and_resume(self, host, tool) -> None:
        host.open_tab(tool.tool_id)
        assert host.pop_out_tab(tool.tool_id) is True
        assert tool.pause_calls == 1
        assert tool.resume_calls == 1
        assert tool.tool_id in host.popped_out_tools()

    def test_dock_back_brackets_with_pause_and_resume(self, host, tool) -> None:
        host.open_tab(tool.tool_id)
        host.pop_out_tab(tool.tool_id)
        index = host.dock_back(tool.tool_id)
        assert index >= 0
        assert tool.pause_calls == 2
        assert tool.resume_calls == 2
        assert tool.tool_id in host.open_tool_ids()

    def test_pop_out_unknown_tool_is_noop(self, host, tool) -> None:
        assert host.pop_out_tab("missing") is False


class TestTabIntrospection:
    def test_open_tool_ids_in_display_order(self, host, tool) -> None:
        other = _HookTool(f"{tool.tool_id}::other")
        register_embeddable_tool(other)
        try:
            host.open_tab(tool.tool_id)
            host.open_tab(other.tool_id)
            assert host.open_tool_ids() == [tool.tool_id, other.tool_id]
        finally:
            EMBEDDABLE_TOOL_REGISTRY.pop(other.tool_id, None)

    def test_active_tool_id_tracks_current_tab(self, host, tool) -> None:
        assert host.active_tool_id() is None
        host.open_tab(tool.tool_id)
        assert host.active_tool_id() == tool.tool_id

    def test_focus_tab_unknown_raises_keyerror(self, host, tool) -> None:
        with pytest.raises(KeyError):
            host.focus_tab("missing")

    def test_focus_tab_resurfaces_backgrounded_tool(self, host, tool) -> None:
        host.open_tab(tool.tool_id)
        assert host.close_tab(tool.tool_id, destroy=False) is True
        assert tool.tool_id in host.backgrounded_tools()
        host.focus_tab(tool.tool_id)
        assert tool.tool_id in host.open_tool_ids()
        assert host.active_tool_id() == tool.tool_id
