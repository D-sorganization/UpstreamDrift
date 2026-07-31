"""Tests for ChatService app_state wiring (issue #5470).

TDD suite: verifies that the Sidekick chat assistant injects the current
application state and historical diagnostic outputs into the AI adapter's
system context.

RED phase: all tests should fail until the implementation lands.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.shared.python.app_state import HistoryStore, StateLogger, agent_context

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chat_service(app_state_provider=None, tmp_path=None):
    """Return a ChatService with a mocked adapter and optional provider."""
    with patch("src.api.services.chat_service.ChatService._load_adapter"):
        from src.api.services.chat_service import ChatService

        svc = ChatService(app_state_provider=app_state_provider)
        if tmp_path is not None:
            svc.PERSIST_DIR = tmp_path / "chat_sessions"
        svc._adapter = MagicMock()
        return svc


def _make_store_with_event() -> HistoryStore:
    """Return a HistoryStore pre-loaded with one test event."""
    store = HistoryStore()
    store.append_event("test_event", {"key": "value"})
    return store


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAppStateProviderInit:
    """Tests for the app_state_provider constructor argument."""

    def test_chat_service_without_provider_accepts_none(self) -> None:
        """ChatService can be constructed without an app_state_provider."""
        svc = _make_chat_service(app_state_provider=None)
        assert svc._app_state_provider is None

    def test_chat_service_with_callable_provider_stores_it(self) -> None:
        """A callable app_state_provider is stored on the service."""

        def provider():
            return {
                "events": [],
                "last_diagnostics": [],
                "summary": "",
            }  # noqa: E731

        svc = _make_chat_service(app_state_provider=provider)
        assert svc._app_state_provider is provider

    def test_invalid_provider_type_raises_type_error(self) -> None:
        """Passing a non-callable, non-None provider raises TypeError."""
        with pytest.raises(TypeError, match="app_state_provider"):
            _make_chat_service(app_state_provider="not-callable")


class TestNoStateContext:
    """Without a provider, no state system message is injected."""

    def test_chat_service_without_provider_sends_no_state_context(
        self, tmp_path
    ) -> None:
        """stream_response should NOT inject a state system message when provider is None."""
        svc = _make_chat_service(app_state_provider=None, tmp_path=tmp_path)
        ctx = svc.get_or_create_session(None)
        svc.add_user_message(ctx.session_id, "hello")

        # Capture the context passed to the adapter
        captured_contexts: list[Any] = []

        def _fake_stream(message, context, tools):
            captured_contexts.append(context)
            return iter([])

        svc._adapter.stream_response.side_effect = _fake_stream

        import asyncio

        async def _collect():
            return [chunk async for chunk in svc.stream_response(ctx.session_id)]

        asyncio.run(_collect())

        # There should be no system message about "application state"
        if captured_contexts:
            messages = captured_contexts[0].messages
            system_messages = [m for m in messages if m.role == "system"]
            state_messages = [
                m for m in system_messages if "application state" in m.content.lower()
            ]
            assert state_messages == []


class TestStateContextInjection:
    """With a provider, state context is injected as a system message."""

    def test_chat_service_with_provider_injects_state_as_system_message(
        self, tmp_path
    ) -> None:
        """stream_response injects a system message from app_state_provider."""
        store = _make_store_with_event()
        provider = lambda: agent_context(store)  # noqa: E731

        svc = _make_chat_service(app_state_provider=provider, tmp_path=tmp_path)
        ctx = svc.get_or_create_session(None)
        svc.add_user_message(ctx.session_id, "hello")

        captured_contexts: list[Any] = []

        def _fake_stream(message, context, tools):
            captured_contexts.append(context)
            return iter([])

        svc._adapter.stream_response.side_effect = _fake_stream

        import asyncio

        async def _collect():
            return [chunk async for chunk in svc.stream_response(ctx.session_id)]

        asyncio.run(_collect())

        assert captured_contexts, "adapter should have been called"
        messages = captured_contexts[0].messages
        system_messages = [m for m in messages if m.role == "system"]
        state_messages = [
            m for m in system_messages if "application state" in m.content.lower()
        ]
        assert (
            state_messages
        ), "Expected at least one 'application state' system message"

    def test_state_context_includes_events_key(self, tmp_path) -> None:
        """Injected state JSON contains the 'events' key."""
        store = _make_store_with_event()
        provider = lambda: agent_context(store)  # noqa: E731

        svc = _make_chat_service(app_state_provider=provider, tmp_path=tmp_path)
        ctx = svc.get_or_create_session(None)
        svc.add_user_message(ctx.session_id, "hello")

        captured_contexts: list[Any] = []

        def _fake_stream(message, context, tools):
            captured_contexts.append(context)
            return iter([])

        svc._adapter.stream_response.side_effect = _fake_stream

        import asyncio

        async def _collect():
            return [chunk async for chunk in svc.stream_response(ctx.session_id)]

        asyncio.run(_collect())

        assert captured_contexts
        state_msg = next(
            m
            for m in captured_contexts[0].messages
            if m.role == "system" and "application state" in m.content.lower()
        )
        # The content must contain JSON with "events" key
        assert '"events"' in state_msg.content

    def test_state_context_includes_last_diagnostics_key(self, tmp_path) -> None:
        """Injected state JSON contains the 'last_diagnostics' key."""
        store = _make_store_with_event()
        provider = lambda: agent_context(store)  # noqa: E731

        svc = _make_chat_service(app_state_provider=provider, tmp_path=tmp_path)
        ctx = svc.get_or_create_session(None)
        svc.add_user_message(ctx.session_id, "hello")

        captured_contexts: list[Any] = []

        def _fake_stream(message, context, tools):
            captured_contexts.append(context)
            return iter([])

        svc._adapter.stream_response.side_effect = _fake_stream

        import asyncio

        async def _collect():
            return [chunk async for chunk in svc.stream_response(ctx.session_id)]

        asyncio.run(_collect())

        assert captured_contexts
        state_msg = next(
            m
            for m in captured_contexts[0].messages
            if m.role == "system" and "application state" in m.content.lower()
        )
        assert '"last_diagnostics"' in state_msg.content

    def test_state_context_is_json_serializable(self, tmp_path) -> None:
        """The dict returned by the provider must be JSON-serializable."""
        store = _make_store_with_event()
        provider = lambda: agent_context(store)  # noqa: E731

        state_dict = provider()
        # Should not raise
        serialized = json.dumps(state_dict)
        parsed = json.loads(serialized)
        assert "events" in parsed
        assert "last_diagnostics" in parsed
        assert "summary" in parsed


class TestProviderResiliency:
    """Provider exceptions must not crash the chat session."""

    def test_provider_exception_does_not_crash_chat(self, tmp_path) -> None:
        """If app_state_provider raises, stream_response continues normally."""
        from unittest.mock import MagicMock

        def _raising_provider() -> dict:
            raise RuntimeError("provider exploded")

        svc = _make_chat_service(
            app_state_provider=_raising_provider, tmp_path=tmp_path
        )
        ctx = svc.get_or_create_session(None)
        svc.add_user_message(ctx.session_id, "hello")

        captured_contexts: list[Any] = []

        def _fake_stream(message, context, tools):
            captured_contexts.append(context)
            chunk = MagicMock()
            chunk.content = "ok"
            chunk.tool_call_delta = None
            return iter([chunk])

        svc._adapter.stream_response.side_effect = _fake_stream

        import asyncio

        async def _collect():
            return [chunk async for chunk in svc.stream_response(ctx.session_id)]

        chunks = asyncio.run(_collect())
        # Chat must still work despite the provider failure
        assert "ok" in chunks
        # No state system message should have been injected (provider failed)
        if captured_contexts:
            messages = captured_contexts[0].messages
            state_messages = [
                m
                for m in messages
                if m.role == "system" and "application state" in m.content.lower()
            ]
            assert state_messages == []


class TestChatEventLogging:
    """Chat events are recorded into app_state (feedback loop)."""

    def test_add_user_message_logs_chat_event(self, tmp_path) -> None:
        """add_user_message records a 'chat.message_sent' event in StateLogger."""
        store = HistoryStore()
        logger = StateLogger(store=store)

        with patch(
            "src.api.services.chat_service.get_state_logger", return_value=logger
        ):
            svc = _make_chat_service(tmp_path=tmp_path)
            ctx = svc.get_or_create_session(None)
            svc.add_user_message(ctx.session_id, "hello from user")

        events = store.snapshot()
        types = [e.type for e in events]
        assert "chat.message_sent" in types

    def test_add_user_message_log_includes_session_id(self, tmp_path) -> None:
        """chat.message_sent payload carries the session_id."""
        store = HistoryStore()
        logger = StateLogger(store=store)

        with patch(
            "src.api.services.chat_service.get_state_logger", return_value=logger
        ):
            svc = _make_chat_service(tmp_path=tmp_path)
            ctx = svc.get_or_create_session(None)
            svc.add_user_message(ctx.session_id, "hello")

        event = next(e for e in store.snapshot() if e.type == "chat.message_sent")
        assert event.payload.get("session_id") == ctx.session_id

    def test_stream_response_logs_response_received(self, tmp_path) -> None:
        """stream_response records a 'chat.response_received' event on success."""
        store = HistoryStore()
        logger = StateLogger(store=store)

        with patch(
            "src.api.services.chat_service.get_state_logger", return_value=logger
        ):
            svc = _make_chat_service(tmp_path=tmp_path)
            ctx = svc.get_or_create_session(None)
            svc.add_user_message(ctx.session_id, "hello")

            svc._adapter.stream_response.return_value = iter([])

            import asyncio

            asyncio.run(_collect_async(svc.stream_response(ctx.session_id)))

        types = [e.type for e in store.snapshot()]
        assert "chat.response_received" in types


async def _collect_async(ait):
    """Consume an async iterator and return all items."""
    return [item async for item in ait]
