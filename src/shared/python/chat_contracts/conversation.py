"""Conversation snapshots shared across chat and memory services."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ArchivedMessage:
    """Minimal archived message shape consumed by memory extraction."""

    role: str
    content: str
    tool_call_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArchivedConversationContext:
    """Dependency-free conversation snapshot for archived chat processing."""

    session_id: str
    messages: list[ArchivedMessage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str, **kwargs: Any) -> ArchivedMessage:
        """Append a message using the same small API as AI ConversationContext."""
        message = ArchivedMessage(role=role, content=content, **kwargs)
        self.messages.append(message)
        return message
