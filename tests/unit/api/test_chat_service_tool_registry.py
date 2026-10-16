"""ChatService must build one ToolRegistry class the Tools registrars accept (UD #12893)."""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.mark.unit
def test_chat_service_registers_golf_suite_tools_without_type_error() -> None:
    """No duplicate ToolRegistry identity between UD and the vendored Tools code."""
    from src.api.services.chat_service import ChatService

    with patch.object(ChatService, "_load_adapter", lambda self: None):
        service = ChatService()

    names = {tool.name for tool in service._tool_registry.list_tools()}
    assert "answer_canonical_core_question" in names
    assert names, "golf suite tools must be registered"


@pytest.mark.unit
def test_sidekick_analytics_registrar_accepts_chat_service_registry() -> None:
    from shared.python.ai.tool_registry import ToolRegistry as ToolsRegistry
    from src.api.services.chat_service import ChatService

    with patch.object(ChatService, "_load_adapter", lambda self: None):
        service = ChatService()

    assert isinstance(service._tool_registry, ToolsRegistry)
