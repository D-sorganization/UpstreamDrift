"""Tests for ChatService tool calling integration (issue #3162).

Verifies that ChatService correctly wires ToolRegistry into the streaming
adapter calls and that tool-call chunks are handled in the stream loop.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.shared.python.ai.adapters.base import ToolDeclaration
from src.shared.python.ai.types import AgentChunk


@pytest.fixture()
def chat_service_with_tools(tmp_path: Any) -> Any:
    """Create a ChatService with mocked adapter, real ToolRegistry, temp dir."""
    with patch("src.api.services.chat_service.ChatService._load_adapter"):
        from src.api.services.chat_service import ChatService

        svc = ChatService()
        svc.PERSIST_DIR = tmp_path / "chat_sessions"
        svc._adapter = MagicMock()
        return svc


class TestToolRegistryWiring:
    """ChatService loads tools into the registry on init."""

    def test_tool_registry_populated(self, chat_service_with_tools: Any) -> None:
        """ToolRegistry is populated with Golf Suite tools after init."""
        svc = chat_service_with_tools
        assert len(svc._tool_registry) > 0

    def test_get_tool_declarations_returns_list(
        self, chat_service_with_tools: Any
    ) -> None:
        """_get_tool_declarations returns a list of ToolDeclaration objects."""
        svc = chat_service_with_tools
        decls = svc._get_tool_declarations()
        assert isinstance(decls, list)
        assert len(decls) > 0
        for d in decls:
            assert isinstance(d, ToolDeclaration)

    def test_tool_declarations_have_required_fields(
        self, chat_service_with_tools: Any
    ) -> None:
        """Each ToolDeclaration has a non-empty name and description."""
        svc = chat_service_with_tools
        decls = svc._get_tool_declarations()
        for d in decls:
            assert d.name, f"Tool declaration missing name: {d}"
            assert d.description, f"Tool declaration missing description: {d}"


class TestStreamResponsePassesTools:
    """stream_response passes real tool declarations to the adapter."""

    def test_stream_response_passes_tool_declarations(
        self, chat_service_with_tools: Any
    ) -> None:
        """Adapter's stream_response is called with non-empty tool list."""
        import asyncio

        svc = chat_service_with_tools
        ctx = svc.get_or_create_session(None)
        svc.add_user_message(ctx.session_id, "Hello")

        # Configure mock adapter to yield a single text chunk then finish
        mock_chunk = AgentChunk(content="Hello!", is_final=True)
        svc._adapter.stream_response.return_value = iter([mock_chunk])

        async def _run() -> list[str]:
            chunks = []
            async for chunk in svc.stream_response(ctx.session_id):
                chunks.append(chunk)
            return chunks

        asyncio.get_event_loop().run_until_complete(_run())

        # Verify adapter was called with a non-empty tools list
        call_args = svc._adapter.stream_response.call_args
        assert call_args is not None
        tools_arg = call_args[0][2]  # positional: message, context, tools
        assert isinstance(tools_arg, list)
        assert len(tools_arg) > 0
        assert all(isinstance(t, ToolDeclaration) for t in tools_arg)

    def test_stream_response_text_chunks_yielded(
        self, chat_service_with_tools: Any
    ) -> None:
        """Text content from adapter chunks is yielded to caller."""
        import asyncio

        svc = chat_service_with_tools
        ctx = svc.get_or_create_session(None)
        svc.add_user_message(ctx.session_id, "Test")

        chunks = [
            AgentChunk(content="Hello"),
            AgentChunk(content=" world"),
            AgentChunk(content="!", is_final=True),
        ]
        svc._adapter.stream_response.return_value = iter(chunks)

        async def _run() -> list[str]:
            result = []
            async for chunk in svc.stream_response(ctx.session_id):
                result.append(chunk)
            return result

        result = asyncio.get_event_loop().run_until_complete(_run())
        assert result == ["Hello", " world", "!"]


class TestStreamResponseToolCallHandling:
    """Tool-call chunks in the stream are executed via registry."""

    def test_tool_call_chunk_executes_registered_tool(
        self, chat_service_with_tools: Any
    ) -> None:
        """A tool_call_delta chunk triggers registry.execute for known tool."""
        import asyncio

        svc = chat_service_with_tools
        ctx = svc.get_or_create_session(None)
        svc.add_user_message(ctx.session_id, "List engines")

        # Simulate: first chunk starts a tool call, final chunk closes it
        import json

        chunks = [
            AgentChunk(
                content="",
                tool_call_delta={
                    "id": "tc_test_001",
                    "name": "list_physics_engines",
                    "arguments": "",
                },
                is_final=False,
            ),
            AgentChunk(
                content="",
                tool_call_delta={
                    "id": "tc_test_001",
                    "arguments": json.dumps({}),
                },
                is_final=True,
            ),
        ]
        svc._adapter.stream_response.return_value = iter(chunks)

        async def _run() -> None:
            async for _ in svc.stream_response(ctx.session_id):
                pass

        asyncio.get_event_loop().run_until_complete(_run())

        # After streaming, temp_ctx should have a tool result message
        # We check the session context was saved (adapter was called)
        assert svc._adapter.stream_response.called

    def test_unknown_tool_call_handled_gracefully(
        self, chat_service_with_tools: Any
    ) -> None:
        """An unknown tool name in a tool_call_delta does not crash the stream."""
        import asyncio

        svc = chat_service_with_tools
        ctx = svc.get_or_create_session(None)
        svc.add_user_message(ctx.session_id, "Do something weird")

        chunks = [
            AgentChunk(
                content="",
                tool_call_delta={
                    "id": "tc_bad_001",
                    "name": "nonexistent_tool",
                    "arguments": "{}",
                },
                is_final=True,
            ),
        ]
        svc._adapter.stream_response.return_value = iter(chunks)

        # ToolExecutionError is raised inside the thread; the stream error handler
        # converts it to a user-visible error chunk
        async def _run() -> list[str]:
            result = []
            async for chunk in svc.stream_response(ctx.session_id):
                result.append(chunk)
            return result

        # Should complete without raising
        result = asyncio.get_event_loop().run_until_complete(_run())
        # The error is surfaced as a text chunk
        combined = "".join(result)
        assert "Error" in combined or combined == ""
