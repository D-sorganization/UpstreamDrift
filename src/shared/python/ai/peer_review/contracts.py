"""Pydantic contracts for the ``ai.peer_review`` package (Tools #2738).

These models define the data boundary between callers, reviewers, the
coordinator, and chat integration. Validation lives here so that misuse
fails loudly at the edge (Design-by-Contract).
"""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Subject kinds the peer-review subsystem knows how to receive. New kinds
# require coordinated changes to the reviewers' prompt templates.
SubjectKind = Literal["message", "code_block", "session_summary"]

# Roles a reviewer may take. Tie-breaker ordering in ``consensus`` depends
# on this list — do not reorder casually.
ReviewerRole = Literal["critic", "advocate", "specialist"]

# Verdicts a reviewer may return. ``abstain`` is treated as a non-vote
# by consensus.
VerdictKind = Literal["approve", "request_changes", "reject", "abstain"]

# Final consensus disposition returned to the caller.
ConsensusKind = Literal["approved", "needs_revision", "rejected", "no_consensus"]


def _default_request_id() -> str:
    """Generate a fresh hex request id for a :class:`ReviewRequest`."""
    return uuid.uuid4().hex


class ReviewRequest(BaseModel):
    """A request to have some subject peer-reviewed.

    Validation enforces ``criteria_set`` is non-empty and that
    ``deadline_seconds`` is positive.
    """

    model_config = ConfigDict(extra="forbid")

    subject_kind: SubjectKind
    subject_id: str = Field(..., min_length=1)
    requester_agent_id: str = Field(..., min_length=1)
    criteria_set: list[str] = Field(...)
    deadline_seconds: float = Field(default=30.0, gt=0.0)
    request_id: str = Field(default_factory=_default_request_id)

    @field_validator("criteria_set")
    @classmethod
    def _criteria_non_empty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("criteria_set must be non-empty")
        return value


class ReviewerDescriptor(BaseModel):
    """Static description of a reviewer agent.

    Stored on the registry and propagated into verdicts so that audit
    sinks can attribute decisions without reaching back into the registry.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_id: str = Field(..., min_length=1)
    provider: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    role: ReviewerRole
    expertise_tags: list[str] = Field(default_factory=list)


class ReviewSubject(BaseModel):
    """The actual content being reviewed.

    Kept separate from :class:`ReviewRequest` so that the request can be
    serialised and stored without the (potentially large) content payload.
    """

    model_config = ConfigDict(extra="forbid")

    kind: SubjectKind
    subject_id: str = Field(..., min_length=1)
    content: str = Field(default="")


class ReviewVerdict(BaseModel):
    """A single reviewer's verdict on a :class:`ReviewSubject`.

    ``reviewer_role`` mirrors the reviewer's descriptor role and is used
    by the consensus tie-breaker without that function needing to consult
    the registry (LOD).
    """

    model_config = ConfigDict(extra="forbid")

    reviewer_agent_id: str = Field(..., min_length=1)
    verdict: VerdictKind
    reasoning: str = ""
    suggested_revisions: list[str] = Field(default_factory=list)
    confidence_0_to_1: float = Field(..., ge=0.0, le=1.0)
    reviewer_role: ReviewerRole = "critic"


class PeerReviewResult(BaseModel):
    """Result returned by :class:`ReviewCoordinator.run_review`.

    ``final_disposition`` is currently identical to ``consensus`` but is
    kept separate so that future policy layers (e.g. requester veto, retry
    budget exhausted) can override the raw consensus.
    """

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(..., min_length=1)
    verdicts: list[ReviewVerdict] = Field(default_factory=list)
    consensus: ConsensusKind
    final_disposition: ConsensusKind
    audit_trail: list[dict] = Field(default_factory=list)


__all__ = [
    "ConsensusKind",
    "PeerReviewResult",
    "ReviewerDescriptor",
    "ReviewerRole",
    "ReviewRequest",
    "ReviewSubject",
    "ReviewVerdict",
    "SubjectKind",
    "VerdictKind",
]
