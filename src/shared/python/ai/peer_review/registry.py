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

from collections.abc import Iterable, Sequence

from .base import Reviewer


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


__all__ = ["ReviewerRegistry"]
