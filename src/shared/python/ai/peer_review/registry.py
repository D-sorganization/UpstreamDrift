"""Reviewer registry (Tools #2738).

A registry maps reviewer agent ids to reviewer instances and knows how to
assemble a panel for a set of criteria.

Panel composition rules (intentionally simple — keep them legible to GUI
and audit consumers):

- All registered critics are always part of every panel.
- All registered advocates are always part of every panel.
- Specialists are included only if any of their ``expertise_tags`` appears
  in the request's ``criteria_set``.

This means a registry that contains only specialists with mismatched tags
will return an empty panel — the coordinator surfaces that as
:class:`InsufficientPanelError`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from .base import Reviewer

if TYPE_CHECKING:
    from ._llm import ReviewerLLMClient

_logger = logging.getLogger(__name__)


def default_llm_client() -> ReviewerLLMClient:
    """Return the default reviewer LLM client for assembling a panel.

    Selects the production :class:`AdapterReviewerLLMClient` when a provider
    adapter is available (best-available resolution via
    :class:`AdapterFactory`), and falls back to the deterministic
    :class:`StubReviewerLLMClient` when running offline / with no configured
    provider.

    This is the wiring point #3177 asks for: callers that build the reviewer
    panel (builtin reviewers, chat integration) obtain their LLM client here
    so real peer review happens whenever a provider is reachable, without the
    panel ever failing to construct when none is.

    The adapter import is performed lazily so the ``peer_review`` package stays
    importable without the adapters subpackage on the path (Orthogonality).
    """
    from ._llm import AdapterReviewerLLMClient, StubReviewerLLMClient

    try:
        from shared.python.ai.adapters.factory import AdapterFactory

        adapter = AdapterFactory.get_best_available()
    except Exception as exc:  # noqa: BLE001 — degrade to stub on any wiring error
        _logger.info(
            "No provider adapter available for peer review (%s); using stub",
            type(exc).__name__,
        )
        adapter = None

    if adapter is None:
        _logger.info("Peer review running offline: using StubReviewerLLMClient")
        return StubReviewerLLMClient()

    _logger.info(
        "Peer review using AdapterReviewerLLMClient backed by %s",
        type(adapter).__name__,
    )
    return AdapterReviewerLLMClient(adapter)


class ReviewerRegistry:
    """In-memory map of ``agent_id`` → :class:`Reviewer`."""

    def __init__(self) -> None:
        self._reviewers: dict[str, Reviewer] = {}

    def register(self, reviewer: Reviewer) -> Reviewer:
        """Register a reviewer. Raises :class:`ValueError` on duplicate id."""
        agent_id = reviewer.descriptor.agent_id
        if agent_id in self._reviewers:
            raise ValueError(f"Reviewer agent_id already registered: {agent_id!r}")
        self._reviewers[agent_id] = reviewer
        return reviewer

    def get(self, agent_id: str) -> Reviewer:
        """Look up a reviewer by id. Raises :class:`KeyError` on miss."""
        return self._reviewers[agent_id]

    def list(self) -> Sequence[Reviewer]:
        """Return all registered reviewers in registration order."""
        return tuple(self._reviewers.values())

    def panel_for(self, criteria: Iterable[str]) -> Sequence[Reviewer]:
        """Build the panel for a set of review criteria.

        See module docstring for the composition rules.
        """
        criteria_set = set(criteria)
        panel: list[Reviewer] = []
        for reviewer in self._reviewers.values():
            role = reviewer.descriptor.role
            if role in ("critic", "advocate"):
                panel.append(reviewer)
            elif role == "specialist":
                tags = set(reviewer.descriptor.expertise_tags)
                if tags & criteria_set:
                    panel.append(reviewer)
        return tuple(panel)


__all__ = ["ReviewerRegistry", "default_llm_client"]
