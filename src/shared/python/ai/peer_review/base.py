"""Reviewer abstract base class (Tools #2738)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .contracts import (
    ReviewerDescriptor,
    ReviewRequest,
    ReviewSubject,
    ReviewVerdict,
)


class Reviewer(ABC):
    """Abstract base for a peer reviewer.

    Subclasses provide an async :meth:`review` and supply a
    :class:`ReviewerDescriptor` to either the constructor or as a class
    attribute. The descriptor is the only metadata the rest of the system
    is permitted to observe (Law of Demeter).
    """

    def __init__(self, *, descriptor: ReviewerDescriptor) -> None:
        if not isinstance(descriptor, ReviewerDescriptor):
            raise TypeError("descriptor must be a ReviewerDescriptor instance")
        self._descriptor = descriptor

    @property
    def descriptor(self) -> ReviewerDescriptor:
        """The reviewer's static descriptor (id, role, provider, model)."""
        return self._descriptor

    @abstractmethod
    async def review(
        self,
        request: ReviewRequest,
        subject: ReviewSubject,
    ) -> ReviewVerdict:
        """Return a verdict for the supplied subject under the request's
        criteria. Implementations should never raise on LLM failures;
        they should return a verdict of ``"abstain"`` with the failure
        captured in ``reasoning``.
        """


__all__ = ["Reviewer"]
