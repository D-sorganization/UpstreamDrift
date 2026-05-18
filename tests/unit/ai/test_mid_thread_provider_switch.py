"""Tests for mid-thread provider switching contract.

Tests the pure-Python behavior of AIAssistantPanel.switch_provider() using
a lightweight mock panel that avoids full Qt widget construction.

DbC invariants verified:
- Pre-condition: a session must be loaded (non-empty context)
- Invariant: message history is never mutated by a switch
- Post-condition: self._adapter points to new provider after switch
- Post-condition: a provider_switched event is recorded in session metadata
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


class _MockPanel:
    """Minimal stand-in for AIAssistantPanel exposing only the switch methods.

    Mirrors the switch_provider() implementation without constructing Qt widgets.
    """

    def __init__(self) -> None:
        from src.shared.python.ai.types import ConversationContext

        self._context = ConversationContext()
        self._adapter: object = None
        self._status_text: str = ""

    def _set_status(self, status: str) -> None:
        self._status_text = status

    def _create_adapter_for_provider(
        self, provider_label: str, model_id: str
    ) -> object:
        # Override in tests
        return None

    def switch_provider(
        self,
        provider_label: str,
        model_id: str,
        thinking_level: str,
    ) -> None:
        """Mirror of AIAssistantPanel.switch_provider() for unit testing."""
        if not self._context.messages:
            raise RuntimeError(
                "No active session: cannot switch provider without a loaded conversation. "
                "Start a new chat or load an existing session first."
            )

        old_provider = (
            self._adapter.capabilities.provider_name if self._adapter else "none"
        )

        new_adapter = self._create_adapter_for_provider(provider_label, model_id)
        if new_adapter is None:
            self._set_status(f"⚠ Could not connect to {provider_label}")
            return

        self._adapter = new_adapter

        from datetime import datetime, timezone

        events: list = self._context.metadata.setdefault("events", [])
        events.append(
            {
                "type": "provider_switched",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "old_provider": old_provider,
                "new_provider": provider_label.lower(),
                "new_model": model_id,
                "thinking_level": thinking_level,
            }
        )

        self._set_status(f"Switched to {provider_label} ({model_id})")


class TestSwitchDoesNotMutateHistory:
    """Switching provider must leave message history unchanged."""

    def test_switch_does_not_mutate_history(self) -> None:
        """After switching, session.messages is identical to pre-switch snapshot."""
        from src.shared.python.ai.types import ConversationContext

        panel = _MockPanel()

        ctx = ConversationContext()
        ctx.add_user_message("Hello")
        ctx.add_assistant_message("Hi there!")
        ctx.add_user_message("Tell me about golf biomechanics.")
        panel._context = ctx

        messages_before = [(m.role, m.content) for m in ctx.messages]

        mock_adapter = MagicMock()
        mock_adapter.capabilities.provider_name = "openai"

        with patch.object(
            panel, "_create_adapter_for_provider", return_value=mock_adapter
        ):
            panel.switch_provider("openai", "gpt-4o", "off")

        messages_after = [(m.role, m.content) for m in panel._context.messages]
        assert (
            messages_before == messages_after
        ), "Message history must not be mutated by provider switch"

    def test_switch_does_not_add_messages_to_history(self) -> None:
        """switch_provider must not append any messages to the history."""
        from src.shared.python.ai.types import ConversationContext

        panel = _MockPanel()

        ctx = ConversationContext()
        ctx.add_user_message("Hello")
        panel._context = ctx
        count_before = len(ctx.messages)

        mock_adapter = MagicMock()
        mock_adapter.capabilities.provider_name = "anthropic"

        with patch.object(
            panel, "_create_adapter_for_provider", return_value=mock_adapter
        ):
            panel.switch_provider("anthropic", "claude-3-5-sonnet-20241022", "off")

        assert len(panel._context.messages) == count_before


class TestSwitchReplaysHistoryToNewAdapter:
    """After switch, the new adapter is stored for the next message."""

    def test_switch_replays_history_to_new_adapter(self) -> None:
        """New adapter is stored as self._adapter after switch_provider()."""
        from src.shared.python.ai.types import ConversationContext

        panel = _MockPanel()

        ctx = ConversationContext()
        ctx.add_user_message("Hello")
        ctx.add_assistant_message("Hi!")
        panel._context = ctx

        mock_new_adapter = MagicMock()
        mock_new_adapter.capabilities.provider_name = "openai"

        with patch.object(
            panel, "_create_adapter_for_provider", return_value=mock_new_adapter
        ):
            panel.switch_provider("openai", "gpt-4o", "off")

        assert (
            panel._adapter is mock_new_adapter
        ), "self._adapter must be the new adapter after switch_provider()"

    def test_failed_switch_does_not_update_adapter(self) -> None:
        """If adapter creation fails, self._adapter is unchanged."""
        from src.shared.python.ai.types import ConversationContext

        panel = _MockPanel()
        original_adapter = MagicMock()
        original_adapter.capabilities.provider_name = "ollama"
        panel._adapter = original_adapter

        ctx = ConversationContext()
        ctx.add_user_message("Hello")
        panel._context = ctx

        with patch.object(panel, "_create_adapter_for_provider", return_value=None):
            panel.switch_provider("openai", "gpt-4o", "off")

        assert (
            panel._adapter is original_adapter
        ), "self._adapter must be unchanged when adapter creation fails"


class TestSwitchRecordsProvenance:
    """switch_provider must record a provider_switched event in metadata."""

    def test_switch_records_provenance_event(self) -> None:
        """context.metadata must contain a provider_switched event after switch."""
        from src.shared.python.ai.types import ConversationContext

        panel = _MockPanel()

        ctx = ConversationContext()
        ctx.add_user_message("Hello")
        panel._context = ctx

        mock_adapter = MagicMock()
        mock_adapter.capabilities.provider_name = "anthropic"
        panel._adapter = MagicMock()
        panel._adapter.capabilities.provider_name = "ollama"

        with patch.object(
            panel, "_create_adapter_for_provider", return_value=mock_adapter
        ):
            panel.switch_provider("anthropic", "claude-3-5-sonnet-20241022", "medium")

        events = panel._context.metadata.get("events", [])
        assert len(events) >= 1
        last_event = events[-1]
        assert last_event["type"] == "provider_switched"
        assert "timestamp" in last_event
        assert last_event["new_provider"] == "anthropic"

    def test_switch_records_old_provider_name(self) -> None:
        """The provenance event includes the old provider name."""
        from src.shared.python.ai.types import ConversationContext

        panel = _MockPanel()

        ctx = ConversationContext()
        ctx.add_user_message("Hello")
        panel._context = ctx

        panel._adapter = MagicMock()
        panel._adapter.capabilities.provider_name = "ollama"

        mock_new_adapter = MagicMock()
        mock_new_adapter.capabilities.provider_name = "openai"

        with patch.object(
            panel, "_create_adapter_for_provider", return_value=mock_new_adapter
        ):
            panel.switch_provider("OpenAI", "gpt-4o", "off")

        events = panel._context.metadata.get("events", [])
        assert len(events) >= 1
        last_event = events[-1]
        assert last_event["old_provider"] == "ollama"

    def test_switch_records_model_and_thinking_level(self) -> None:
        """The provenance event includes new_model and thinking_level."""
        from src.shared.python.ai.types import ConversationContext

        panel = _MockPanel()

        ctx = ConversationContext()
        ctx.add_user_message("Hello")
        panel._context = ctx

        mock_adapter = MagicMock()
        mock_adapter.capabilities.provider_name = "anthropic"

        with patch.object(
            panel, "_create_adapter_for_provider", return_value=mock_adapter
        ):
            panel.switch_provider("anthropic", "claude-3-5-sonnet-20241022", "high")

        events = panel._context.metadata.get("events", [])
        last_event = events[-1]
        assert last_event["new_model"] == "claude-3-5-sonnet-20241022"
        assert last_event["thinking_level"] == "high"


class TestSwitchPreconditionNoSession:
    """switch_provider must raise RuntimeError when no messages in context."""

    def test_switch_precondition_no_session_raises(self) -> None:
        """Calling switch_provider with empty context raises RuntimeError."""
        from src.shared.python.ai.types import ConversationContext

        panel = _MockPanel()
        panel._context = ConversationContext()  # empty — no messages

        with pytest.raises(RuntimeError, match="No active session"):
            panel.switch_provider("anthropic", "claude-sonnet-4-6", "high")

    def test_switch_precondition_with_messages_does_not_raise(self) -> None:
        """With at least one message in context, switch_provider must not raise."""
        from src.shared.python.ai.types import ConversationContext

        panel = _MockPanel()

        ctx = ConversationContext()
        ctx.add_user_message("Hello")
        panel._context = ctx

        mock_adapter = MagicMock()
        mock_adapter.capabilities.provider_name = "openai"

        with patch.object(
            panel, "_create_adapter_for_provider", return_value=mock_adapter
        ):
            # Must not raise
            panel.switch_provider("openai", "gpt-4o", "off")
