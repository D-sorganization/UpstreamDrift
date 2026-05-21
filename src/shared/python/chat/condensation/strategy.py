"""Condensation strategies (Tools issue #2736).

Three concrete strategies, all derived from :class:`CondensationStrategy`:

* :class:`KeepRecentStrategy` -- drop everything older than the last
  ``keep_last_n`` messages.
* :class:`SemanticSummaryStrategy` -- collapse older messages into a
  single anchor message. A real summariser can be injected via the
  :class:`SummaryProvider` protocol; the default uses a truncation-based
  placeholder that is deterministic and LLM-free.
* :class:`PinnedAnchorStrategy` -- preserve every message with
  ``metadata["pin"] is True`` plus the recent tail.

Strategies are pure: they accept a :class:`ChatSession` and return a new
:class:`ChatSession` without mutating the input.
"""

from __future__ import annotations

import abc
import dataclasses
from typing import Protocol

from chat.service_base import ChatMessage, ChatSession

from .contracts import CondensationRequest


class SummaryProvider(Protocol):
    """Optional summariser injected into :class:`SemanticSummaryStrategy`."""

    def summarise(self, messages: list[ChatMessage]) -> str: ...


class _TruncationSummary:
    """Deterministic LLM-free fallback summariser.

    Concatenates the first 40 characters of each message body, joined by
    semicolons, capped at 400 characters. Good enough as a structural
    placeholder; production callers can inject a real summariser.
    """

    def summarise(self, messages: list[ChatMessage]) -> str:
        chunks: list[str] = []
        for m in messages:
            head = m.content.strip().splitlines()[0] if m.content else ""
            chunks.append(f"{m.role}: {head[:40]}")
        summary = "; ".join(chunks)
        return summary[:400]


def _clone_session(session: ChatSession, messages: list[ChatMessage]) -> ChatSession:
    """Return a new :class:`ChatSession` with the same id/metadata but a
    deep-copied message list."""
    return ChatSession(
        session_id=session.session_id,
        messages=[dataclasses.replace(m, metadata=dict(m.metadata)) for m in messages],
        metadata=dict(session.metadata),
        created_at=session.created_at,
    )


class CondensationStrategy(abc.ABC):
    """Abstract base class for condensation strategies."""

    name: str = "abstract"

    @abc.abstractmethod
    def apply(self, session: ChatSession, request: CondensationRequest) -> ChatSession:
        """Return a new condensed session. Must not mutate ``session``."""


class KeepRecentStrategy(CondensationStrategy):
    """Drop everything older than the most-recent ``keep_last_n`` messages."""

    name = "keep_recent"

    def apply(self, session: ChatSession, request: CondensationRequest) -> ChatSession:
        if request.keep_last_n < 1:
            raise ValueError("keep_last_n must be >= 1")
        tail = list(session.messages[-request.keep_last_n :])
        return _clone_session(session, tail)


class PinnedAnchorStrategy(CondensationStrategy):
    """Preserve pinned messages + the recent tail; drop the rest."""

    name = "pinned_anchor"

    def apply(self, session: ChatSession, request: CondensationRequest) -> ChatSession:
        if request.keep_last_n < 1:
            raise ValueError("keep_last_n must be >= 1")
        all_msgs = list(session.messages)
        tail = all_msgs[-request.keep_last_n :]
        tail_ids = {id(m) for m in tail}
        pinned = [
            m
            for m in all_msgs[: -request.keep_last_n]
            if m.metadata.get("pin") is True and id(m) not in tail_ids
        ]
        kept = pinned + tail
        return _clone_session(session, kept)


class SemanticSummaryStrategy(CondensationStrategy):
    """Collapse older messages into a single ``Earlier in conversation`` anchor."""

    name = "semantic_summary"

    def __init__(self, summariser: SummaryProvider | None = None) -> None:
        self._summariser: SummaryProvider = summariser or _TruncationSummary()

    def apply(self, session: ChatSession, request: CondensationRequest) -> ChatSession:
        if request.keep_last_n < 1:
            raise ValueError("keep_last_n must be >= 1")
        all_msgs = list(session.messages)
        if len(all_msgs) <= request.keep_last_n:
            # Nothing to summarise; return a clean clone.
            return _clone_session(session, all_msgs)

        older = all_msgs[: -request.keep_last_n]
        tail = all_msgs[-request.keep_last_n :]
        summary_text = self._summariser.summarise(older)
        anchor = ChatMessage(
            role="system",
            content=f"[Earlier in conversation: {summary_text}]",
            metadata={"condensation_anchor": True, "pin": True},
        )
        return _clone_session(session, [anchor] + tail)


STRATEGY_REGISTRY: dict[str, type[CondensationStrategy]] = {
    KeepRecentStrategy.name: KeepRecentStrategy,
    PinnedAnchorStrategy.name: PinnedAnchorStrategy,
    SemanticSummaryStrategy.name: SemanticSummaryStrategy,
}


__all__ = [
    "CondensationStrategy",
    "KeepRecentStrategy",
    "PinnedAnchorStrategy",
    "SemanticSummaryStrategy",
    "SummaryProvider",
    "STRATEGY_REGISTRY",
]
