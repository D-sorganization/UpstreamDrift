"""Ambiguity semantics for club-only candidate sets (CO-02 #10606)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

from src.shared.python.motion_matching.club_only.profiles import ClubOnlyProfile


class AmbiguityStatus(str, Enum):
    """Outcome of observable-ambiguity assessment for a candidate set."""

    UNIQUE = "unique"
    AMBIGUOUS = "ambiguous"
    INFEASIBLE = "infeasible"


@dataclass(frozen=True)
class CandidateScore:
    """Measured residual plus prior score for one body/control candidate."""

    candidate_id: str
    measured_residual_m: float
    prior_score: float
    body_configuration_hash: str
    closure_residual_m: float | None = None
    contact_feasible: bool = True
    claims_force_measurement: bool = False

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must be non-empty")
        if not self.body_configuration_hash:
            raise ValueError("body_configuration_hash must be non-empty")
        if (
            self.measured_residual_m != self.measured_residual_m
            or self.measured_residual_m < 0.0
        ):
            raise ValueError("measured_residual_m must be finite and >= 0")
        if self.prior_score != self.prior_score or not 0.0 <= self.prior_score <= 1.0:
            raise ValueError("prior_score must be finite in [0, 1]")
        if self.closure_residual_m is not None and (
            self.closure_residual_m != self.closure_residual_m
            or self.closure_residual_m < 0.0
        ):
            raise ValueError("closure_residual_m must be finite and >= 0 when set")


@dataclass(frozen=True)
class AmbiguityVerdict:
    """Retained candidates and ambiguity status under a frozen profile."""

    status: AmbiguityStatus
    retained_candidate_ids: frozenset[str]
    required_diverse_candidates: int
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "retained_candidate_ids": sorted(self.retained_candidate_ids),
            "required_diverse_candidates": self.required_diverse_candidates,
            "reason": self.reason,
        }


def assess_ambiguity(
    candidates: Sequence[CandidateScore],
    profile: ClubOnlyProfile,
) -> AmbiguityVerdict:
    """Assess whether identical club residuals leave distinct bodies ambiguous.

    Prior score alone cannot collapse distinct body hashes when measured
    residuals are within the profile ambiguity tolerance.
    """
    if not candidates:
        raise ValueError("candidates must be non-empty")
    required = profile.plausibility.min_diverse_candidates
    tol = profile.plausibility.ambiguity_residual_tol_m

    feasible = tuple(
        c
        for c in candidates
        if c.contact_feasible
        and (
            c.closure_residual_m is None
            or c.closure_residual_m <= profile.physical.max_closure_residual_m
        )
    )
    if not feasible:
        return AmbiguityVerdict(
            status=AmbiguityStatus.INFEASIBLE,
            retained_candidate_ids=frozenset(),
            required_diverse_candidates=required,
            reason="no contact/closure-feasible candidates",
        )

    best_measured = min(c.measured_residual_m for c in feasible)
    near_best = tuple(
        c for c in feasible if abs(c.measured_residual_m - best_measured) <= tol
    )
    body_hashes = {c.body_configuration_hash for c in near_best}
    retained = frozenset(c.candidate_id for c in near_best)

    if len(body_hashes) >= required:
        return AmbiguityVerdict(
            status=AmbiguityStatus.AMBIGUOUS,
            retained_candidate_ids=retained,
            required_diverse_candidates=required,
            reason=(
                "distinct body configurations share measured club residual "
                "within ambiguity tolerance; priors cannot unique them"
            ),
        )

    if len(near_best) == 1:
        return AmbiguityVerdict(
            status=AmbiguityStatus.UNIQUE,
            retained_candidate_ids=retained,
            required_diverse_candidates=required,
            reason="single feasible body hash at best measured residual",
        )

    # Same body family: allow prior only as a secondary rank inside the family.
    ranked = sorted(
        near_best,
        key=lambda c: (-c.prior_score, c.candidate_id),
    )
    return AmbiguityVerdict(
        status=AmbiguityStatus.UNIQUE,
        retained_candidate_ids=frozenset({ranked[0].candidate_id}),
        required_diverse_candidates=required,
        reason="same body-hash family; prior ranks within measured tie",
    )
