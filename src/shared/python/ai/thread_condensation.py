"""Thread condensation and token count visibility (Tools #2736).

Provides:
- :class:`SummaryMessage` — a condensed snapshot of prior conversation turns.
- :func:`condense_thread` — calls an adapter to summarise a message list and
  returns the summary block plus the new active context.
- :func:`estimate_token_count` — rough heuristic (total chars / 4) for UI display.

Design:
- DbC: all public functions validate inputs and raise :exc:`ValueError` /
  :exc:`TypeError` for bad arguments.
- LOD: no method chains deeper than two levels.
- The original message list is never mutated; raw history is preserved for undo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .adapters.base import BaseAgentAdapter

from .types import ConversationContext, Message

UTC = timezone.utc  # noqa: UP017

_DEFAULT_KEEP_RECENT: int = 6

_CONDENSE_SYSTEM_PROMPT = """\
You are a conversation summariser. Given a conversation history, extract and \
return a concise summary structured as:

**Current objective:** <what the user is trying to accomplish>
**Decisions made:** <key decisions or conclusions reached so far>
**Key context:** <essential facts, constraints, or state the assistant must remember>

Be brief — the summary replaces prior messages in the active context window. \
Omit pleasantries and irrelevant detail.\
"""


@dataclass
class SummaryMessage:
    """A condensed snapshot of earlier conversation turns.

    Attributes:
        content: The summarised text produced by the LLM.
        source_count: How many original messages were condensed.
        created_at: When the summary was created [UTC].
        metadata: Arbitrary extra data (e.g. model name, token counts).
    """

    content: str
    source_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_message(self) -> Message:
        """Convert to a :class:`Message` with role ``"system"`` for adapter context."""
        return Message(
            role="system",
            content=f"[Conversation Summary]\n{self.content}",
            timestamp=self.created_at,
            metadata={"is_summary": True, "source_count": self.source_count},
        )


def condense_thread(
    messages: list[Message],
    adapter: BaseAgentAdapter,
    *,
    keep_recent: int = _DEFAULT_KEEP_RECENT,
) -> tuple[SummaryMessage, list[Message]]:
    """Summarise earlier conversation messages using *adapter*.

    The *messages* list is **not** mutated; callers may keep it as the raw
    history buffer for undo.

    Args:
        messages: Full conversation history to condense. Must be a ``list``.
        adapter: An adapter instance whose ``send_message`` method is used for
            the summarisation call.
        keep_recent: How many of the most-recent messages to retain verbatim
            in the returned active context. Defaults to
            :data:`_DEFAULT_KEEP_RECENT`.

    Returns:
        A ``(SummaryMessage, active_context)`` tuple where *active_context*
        is ``[summary] + messages[-keep_recent:]``.

    Raises:
        TypeError: If *messages* is not a ``list``.
        ValueError: If *adapter* is ``None``.
    """
    if not isinstance(messages, list):
        raise TypeError(f"messages must be a list, got {type(messages).__name__!r}")
    if adapter is None:
        raise ValueError("adapter must be provided")

    if not messages:
        summary = SummaryMessage(content="", source_count=0)
        return summary, [summary]

    history_text = _format_history_for_summary(messages)

    if history_text.strip():
        condense_context = ConversationContext()
        condense_context.add_message("system", _CONDENSE_SYSTEM_PROMPT)

        response = adapter.send_message(
            history_text,
            condense_context,
            [],
        )
        summary_content: str = response.content if response.content else ""
    else:
        summary_content = ""

    summary = SummaryMessage(content=summary_content, source_count=len(messages))

    tail = messages[-keep_recent:] if keep_recent > 0 else []
    active: list[Message] = [summary, *tail]

    return summary, active


def estimate_token_count(messages: list[Message]) -> int:
    """Estimate the token count for a list of messages.

    Uses the rough heuristic: ``total_characters // 4``, consistent with the
    heuristic already used in :class:`~src.shared.python.ai.types.ConversationContext`.

    Args:
        messages: List of :class:`Message` objects to estimate.

    Returns:
        Estimated token count as a non-negative integer.
    """
    if not messages:
        return 0
    total_chars = sum(len(m.content) for m in messages)
    return total_chars // 4


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _format_history_for_summary(messages: list[Message]) -> str:
    """Render *messages* as a plain-text transcript for the summarisation prompt."""
    lines: list[str] = []
    for msg in messages:
        role = msg.role.upper()
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)
