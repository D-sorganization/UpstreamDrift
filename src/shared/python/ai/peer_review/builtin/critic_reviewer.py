"""Critic reviewer — looks for problems and missing evidence (Tools #2738)."""

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


class CriticReviewer(Reviewer):
    """Reference critic reviewer.

    The role bias (look for problems) lives in the prompt the LLM client
    builds for ``role="critic"``; this class is provider-agnostic.
    """

    def __init__(
        self,
        *,
        llm_client: ReviewerLLMClient,
        agent_id: str | None = None,
        provider: str = "stub",
        model: str = "stub-1",
    ) -> None:
        descriptor = ReviewerDescriptor(
            agent_id=agent_id or f"critic-{uuid.uuid4().hex[:8]}",
            provider=provider,
            model=model,
            role="critic",
            expertise_tags=[],
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


__all__ = ["CriticReviewer", "StubReviewerLLMClient"]
