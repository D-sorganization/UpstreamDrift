"""EmbeddedHostWidget launcher-context integration tests (#7210)."""

from __future__ import annotations

import os
from typing import Any

import pytest

pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtWidgets import QApplication, QLabel, QWidget  # noqa: E402

from src.launchers.embedded_host import EmbeddedHostWidget  # noqa: E402
from src.shared.python.launcher_embed import (  # noqa: E402
    EmbedCapabilities,
    register_embeddable_tool,
    unregister_embeddable_tool,
)
from src.shared.python.launcher_embed.context import (  # noqa: E402
    LauncherContext,
)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = pytest.mark.unit


class _ContextTool:
    def __init__(self, tool_id: str) -> None:
        self.tool_id = tool_id
        self.contexts: list[LauncherContext] = []
        self.observed: list[dict[str, object]] = []
        self.cleanup_calls = 0

    def embed_capabilities(self) -> EmbedCapabilities:
        return EmbedCapabilities()

    def create_main_widget(self, parent: Any) -> QWidget:
        return QLabel(self.tool_id, parent)

    def cleanup(self) -> None:
        self.cleanup_calls += 1

    def is_dirty(self) -> bool:
        return False

    def set_launcher_context(self, context: LauncherContext) -> None:
        self.contexts.append(context)
        context.subscribe("value_changed:shared_model", self.observed.append)


class _LegacyTool:
    def __init__(self, tool_id: str) -> None:
        self.tool_id = tool_id

    def embed_capabilities(self) -> EmbedCapabilities:
        return EmbedCapabilities()

    def create_main_widget(self, parent: Any) -> QWidget:
        return QLabel(self.tool_id, parent)

    def cleanup(self) -> None:
        pass

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


@pytest.fixture(autouse=True)
def _clean_registry():
    registered: list[str] = []
    yield registered
    for tool_id in registered:
        unregister_embeddable_tool(tool_id)


def _register(tool: Any, registered: list[str]) -> None:
    register_embeddable_tool(tool)
    registered.append(tool.tool_id)


def test_host_injects_one_shared_context_into_opt_in_tools(
    host: EmbeddedHostWidget, _clean_registry: list[str]
) -> None:
    producer = _ContextTool("producer")
    observer = _ContextTool("observer")
    _register(producer, _clean_registry)
    _register(observer, _clean_registry)

    host.open_tab("producer")
    host.open_tab("observer")
    host.launcher_context.set_value("shared_model", "demo.urdf")

    assert producer.contexts == [host.launcher_context]
    assert observer.contexts == [host.launcher_context]
    assert observer.observed[-1]["value"] == "demo.urdf"
    assert producer.observed[-1]["key"] == "shared_model"


def test_legacy_tool_without_context_hook_still_opens(
    host: EmbeddedHostWidget, _clean_registry: list[str]
) -> None:
    tool = _LegacyTool("legacy")
    _register(tool, _clean_registry)

    index = host.open_tab("legacy")

    assert index == 0
    assert host.open_tool_ids() == ["legacy"]


def test_tab_open_and_close_emit_context_events(
    host: EmbeddedHostWidget, _clean_registry: list[str]
) -> None:
    tool = _ContextTool("contextual")
    _register(tool, _clean_registry)
    events: list[tuple[str, dict[str, object]]] = []
    host.launcher_context.subscribe(
        "tab.opened", lambda payload: events.append(("open", payload))
    )
    host.launcher_context.subscribe(
        "tab.closed", lambda payload: events.append(("close", payload))
    )

    host.open_tab("contextual")
    host.close_tab("contextual", destroy=True)

    assert events == [
        ("open", {"tool_id": "contextual", "surface": "tab"}),
        (
            "close",
            {
                "tool_id": "contextual",
                "surface": "tab",
                "destroyed": True,
            },
        ),
    ]


def test_background_close_emits_non_destructive_tab_closed_event(
    host: EmbeddedHostWidget, _clean_registry: list[str]
) -> None:
    tool = _ContextTool("contextual")
    _register(tool, _clean_registry)
    closed: list[dict[str, object]] = []
    host.launcher_context.subscribe("tab.closed", closed.append)

    host.open_tab("contextual")
    host.close_tab("contextual", destroy=False)

    assert closed == [
        {
            "tool_id": "contextual",
            "surface": "tab",
            "destroyed": False,
        }
    ]
