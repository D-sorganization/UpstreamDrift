"""Tests for the headless AssistantSession (assistant_core.py).

Covers:
- Session creation with a mock adapter
- Streaming message handling
- Tool injection interface
- Confirmation-gated tool behaviour
- RAG store injection
- History management and reset
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.shared.python.ai.adapters.base import BaseAgentAdapter, ToolDeclaration
from src.shared.python.ai.assistant_core import AssistantSession
from src.shared.python.ai.tool_registry import ToolRegistry
from src.shared.python.ai.types import (
    AgentChunk,
    AgentResponse,
    ConversationContext,
    ProviderCapabilities,
    ProviderCapability,
)

# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------


class _FakeAdapter(BaseAgentAdapter):
    """Minimal adapter that streams pre-set chunks synchronously."""

    def __init__(self, chunks: list[str] | None = None) -> None:
        self._chunks = chunks or ["Hello", ", world", "!"]

    def send_message(
        self,
        message: str,
        context: ConversationContext,
        tools: list[ToolDeclaration],
    ) -> AgentResponse:
        return AgentResponse(content="".join(self._chunks))

    def stream_response(
        self,
        message: str,
        context: ConversationContext,
        tools: list[ToolDeclaration],
    ) -> Iterator[AgentChunk]:
        for i, text in enumerate(self._chunks):
            is_last = i == len(self._chunks) - 1
            yield AgentChunk(content=text, is_final=is_last)

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supported=frozenset({ProviderCapability.STREAMING}),
            max_tokens=8192,
            model_name="fake",
            provider_name="fake",
        )

    def validate_connection(self) -> tuple[bool, str]:
        return True, "ok"


@pytest.fixture
def fake_adapter() -> _FakeAdapter:
    return _FakeAdapter()


@pytest.fixture
def session(fake_adapter: _FakeAdapter) -> AssistantSession:
    return AssistantSession(adapter=fake_adapter)


# ---------------------------------------------------------------------------
# Session creation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_session_creation_with_injected_adapter(fake_adapter: _FakeAdapter) -> None:
    """Session should accept a pre-built adapter without hitting provider builders."""
    sess = AssistantSession(adapter=fake_adapter)
    assert sess._adapter is fake_adapter
    assert sess._tool_registry is None
    assert sess._rag_store is None


@pytest.mark.unit
def test_session_creation_with_provider_string() -> None:
    """Session should build an OllamaAdapter when provider='ollama' is given."""
    with patch("src.shared.python.ai.assistant_core._build_adapter") as mock_build:
        mock_build.return_value = MagicMock()
        sess = AssistantSession(
            provider="ollama",
            adapter_kwargs={"host": "http://localhost:11434"},
        )
        mock_build.assert_called_once_with("ollama", host="http://localhost:11434")
        assert sess._adapter is mock_build.return_value


@pytest.mark.unit
def test_session_custom_system_prompt(fake_adapter: _FakeAdapter) -> None:
    """Custom system_prompt should override the default."""
    custom = "You are a custom bot."
    sess = AssistantSession(adapter=fake_adapter, system_prompt=custom)
    assert sess._system_prompt == custom


@pytest.mark.unit
def test_build_adapter_unknown_provider() -> None:
    """_build_adapter must raise ValueError for unknown provider names."""
    from src.shared.python.ai.assistant_core import _build_adapter

    with pytest.raises(ValueError, match="Unknown provider"):
        _build_adapter("nonexistent_provider")


# ---------------------------------------------------------------------------
# Streaming message handling
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_send_message_streams_chunks(session: AssistantSession) -> None:
    """send_message should yield the chunks produced by the adapter."""

    async def _run() -> list[str]:
        chunks: list[str] = []
        async for chunk in session.send_message("Hello"):
            chunks.append(chunk)
        return chunks

    result = asyncio.run(_run())
    assert result == ["Hello", ", world", "!"]


@pytest.mark.unit
def test_send_message_saves_to_history(session: AssistantSession) -> None:
    """After streaming, both user and assistant messages should appear in history."""

    async def _run() -> None:
        async for _ in session.send_message("ping"):
            pass

    asyncio.run(_run())

    history = session.get_history()
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[0].content == "ping"
    assert history[1].role == "assistant"
    assert "Hello" in history[1].content


@pytest.mark.unit
def test_send_message_empty_raises(session: AssistantSession) -> None:
    """send_message should raise ValueError for empty or whitespace-only input."""

    async def _run() -> None:
        async for _ in session.send_message("   "):
            pass

    with pytest.raises(ValueError, match="non-empty"):
        asyncio.run(_run())


@pytest.mark.unit
def test_send_message_accumulates_across_turns(session: AssistantSession) -> None:
    """Multiple send_message calls should accumulate conversation history."""

    async def _run() -> None:
        async for _ in session.send_message("turn 1"):
            pass
        async for _ in session.send_message("turn 2"):
            pass

    asyncio.run(_run())

    history = session.get_history()
    roles = [m.role for m in history]
    assert roles == ["user", "assistant", "user", "assistant"]


# ---------------------------------------------------------------------------
# History and reset
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_get_history_returns_copy(session: AssistantSession) -> None:
    """get_history should return an independent copy, not the live list."""

    async def _run() -> None:
        async for _ in session.send_message("hi"):
            pass

    asyncio.run(_run())

    h1 = session.get_history()
    h1.clear()
    h2 = session.get_history()
    assert len(h2) == 2  # original list unaffected


@pytest.mark.unit
def test_reset_clears_history(session: AssistantSession) -> None:
    """reset() should wipe all messages from the history."""

    async def _run() -> None:
        async for _ in session.send_message("before reset"):
            pass

    asyncio.run(_run())
    assert len(session.get_history()) == 2

    session.reset()
    assert session.get_history() == []


# ---------------------------------------------------------------------------
# Tool injection interface
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_tool_registry_is_injected(fake_adapter: _FakeAdapter) -> None:
    """Session should store a provided ToolRegistry."""
    registry = ToolRegistry()
    sess = AssistantSession(adapter=fake_adapter, tool_registry=registry)
    assert sess._tool_registry is registry


@pytest.mark.unit
def test_tool_declarations_passed_to_adapter() -> None:
    """Adapter.stream_response should receive ToolDeclaration objects from registry."""
    registry = ToolRegistry()

    @registry.register(name="ping", description="Returns pong")
    def _ping() -> str:  # type: ignore[return]
        return "pong"

    captured: list[Any] = []

    class CapturingAdapter(_FakeAdapter):
        def stream_response(
            self,
            message: str,
            context: ConversationContext,
            tools: list[ToolDeclaration],
        ) -> Iterator[AgentChunk]:
            captured.extend(tools)
            yield AgentChunk(content="ok", is_final=True)

    sess = AssistantSession(adapter=CapturingAdapter(), tool_registry=registry)

    async def _run() -> None:
        async for _ in sess.send_message("test"):
            pass

    asyncio.run(_run())

    assert len(captured) == 1
    assert captured[0].name == "ping"
    assert captured[0].description == "Returns pong"


@pytest.mark.unit
def test_tool_registry_declarations_method() -> None:
    """ToolRegistry.declarations() must return ToolDeclaration instances."""
    from src.shared.python.ai.adapters.base import ToolDeclaration

    registry = ToolRegistry()

    @registry.register(name="greet", description="Say hello")
    def _greet(name: str) -> str:  # type: ignore[return]
        return f"Hello {name}"

    decls = registry.declarations()
    assert len(decls) == 1
    assert isinstance(decls[0], ToolDeclaration)
    assert decls[0].name == "greet"
    assert "name" in decls[0].parameters


# ---------------------------------------------------------------------------
# Confirmation-gated tools
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_confirmation_required_tool_skipped_without_callback() -> None:
    """Tools with requires_confirmation=True should be skipped if no callback set."""
    registry = ToolRegistry()

    @registry.register(
        name="danger",
        description="Dangerous op",
        requires_confirmation=True,
    )
    def _danger() -> str:  # type: ignore[return]
        return "boom"

    tool_call_delta = {
        "tool_calls": [
            {
                "index": 0,
                "id": "call_1",
                "function": {"name": "danger", "arguments": "{}"},
            }
        ]
    }

    chunks_with_tool: list[AgentChunk] = [
        AgentChunk(content="thinking", is_final=False),
        AgentChunk(
            content="",
            tool_call_delta=tool_call_delta,
            is_final=True,
        ),
    ]

    class ToolCallAdapter(_FakeAdapter):
        def __init__(self) -> None:
            super().__init__()
            self._chunks = []
            self._tool_chunks = chunks_with_tool

        def stream_response(
            self,
            message: str,
            context: ConversationContext,
            tools: list[ToolDeclaration],
        ) -> Iterator[AgentChunk]:
            yield from self._tool_chunks

    sess = AssistantSession(
        adapter=ToolCallAdapter(),
        tool_registry=registry,
        confirmation_callback=None,
    )

    collected: list[str] = []

    async def _run() -> None:
        async for chunk in sess.send_message("go"):
            collected.append(chunk)

    asyncio.run(_run())

    full = "".join(collected)
    assert "skipped" in full.lower() or "confirmation required" in full.lower()


@pytest.mark.unit
def test_confirmation_required_tool_approved_via_callback() -> None:
    """Tool should execute when confirmation_callback returns True."""
    registry = ToolRegistry()
    executed: list[str] = []

    @registry.register(
        name="safe_action",
        description="Do something",
        requires_confirmation=True,
    )
    def _action() -> str:
        executed.append("ran")
        return "done"

    tool_call_delta = {
        "tool_calls": [
            {
                "index": 0,
                "id": "call_2",
                "function": {"name": "safe_action", "arguments": "{}"},
            }
        ]
    }

    chunks_with_tool: list[AgentChunk] = [
        AgentChunk(content="", tool_call_delta=tool_call_delta, is_final=True),
    ]

    class ActionAdapter(_FakeAdapter):
        def stream_response(
            self,
            message: str,
            context: ConversationContext,
            tools: list[ToolDeclaration],
        ) -> Iterator[AgentChunk]:
            yield from chunks_with_tool

    async def _approve(tool_name: str, arguments: dict[str, Any]) -> bool:
        return True

    sess = AssistantSession(
        adapter=ActionAdapter(),
        tool_registry=registry,
        confirmation_callback=_approve,
    )

    async def _run() -> None:
        async for _ in sess.send_message("go"):
            pass

    asyncio.run(_run())
    assert "ran" in executed


# ---------------------------------------------------------------------------
# RAG store injection
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_rag_store_queried_on_send(fake_adapter: _FakeAdapter) -> None:
    """When a rag_store is provided it should be queried for each message."""
    mock_rag = MagicMock()
    mock_doc = MagicMock()
    mock_doc.id = "doc1"
    mock_doc.content = "RAG content here"
    mock_doc.metadata = {"source": "test.md"}
    mock_rag.query.return_value = [(mock_doc, 0.9)]

    sess = AssistantSession(adapter=fake_adapter, rag_store=mock_rag)

    async def _run() -> None:
        async for _ in sess.send_message("help"):
            pass

    asyncio.run(_run())

    mock_rag.query.assert_called_once()


@pytest.mark.unit
def test_rag_store_failure_does_not_crash(fake_adapter: _FakeAdapter) -> None:
    """A failing RAG store should be handled gracefully, not bubble up."""
    mock_rag = MagicMock()
    mock_rag.query.side_effect = RuntimeError("RAG down")

    sess = AssistantSession(adapter=fake_adapter, rag_store=mock_rag)

    async def _run() -> list[str]:
        chunks: list[str] = []
        async for chunk in sess.send_message("hello"):
            chunks.append(chunk)
        return chunks

    result = asyncio.run(_run())
    assert result  # streaming still worked despite RAG failure
