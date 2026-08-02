"""Specialist reviewer — opinionated on a specific expertise (Tools #2738)."""

from __future__ import annotations

import uuid

from .._llm import ReviewerLLMClient, StubReviewerLLMClient
from ..base import Reviewer
from ..contracts import (
    ReviewerDescriptor,
    ReviewRequest,
    ReviewSubject,
    ReviewVerdict,
)
from ._common import evaluate_to_verdict


class SpecialistReviewer(Reviewer):
    """Reference specialist reviewer.

    Specialists must declare at least one non-empty expertise tag — that
    is how the registry decides whether they belong on a panel.
    """

    def __init__(
        self,
        *,
        llm_client: ReviewerLLMClient,
        expertise_tags: list[str],
        agent_id: str | None = None,
        provider: str = "stub",
        model: str = "stub-1",
    ) -> None:
        if not expertise_tags:
            raise ValueError("SpecialistReviewer requires at least one expertise tag")
        descriptor = ReviewerDescriptor(
            agent_id=agent_id or f"specialist-{uuid.uuid4().hex[:8]}",
            provider=provider,
            model=model,
            role="specialist",
            expertise_tags=list(expertise_tags),
        )
        super().__init__(descriptor=descriptor)
        self._llm = llm_client

    async def review(
        self,
        request: ReviewRequest,
        subject: ReviewSubject,
    ) -> ReviewVerdict:
        return await evaluate_to_verdict(
            llm_client=self._llm,
            descriptor=self.descriptor,
            criteria_set=list(request.criteria_set),
            subject_content=subject.content,
        )


__all__ = ["SpecialistReviewer", "StubReviewerLLMClient"]
