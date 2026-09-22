"""Shared native G1 qualification claim invariants for club-only packages."""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["validate_native_g1_claim_contract"]


def validate_native_g1_claim_contract(
    *,
    native_g1_pass: bool,
    claims_native_qualification: bool,
    qualification_blockers: Sequence[str],
) -> None:
    """Fail closed when native pass/claim/blocker fields disagree."""
    if native_g1_pass and not claims_native_qualification:
        raise ValueError("native_g1_pass requires claims_native_qualification")
    if native_g1_pass and qualification_blockers:
        raise ValueError("native_g1_pass cannot retain qualification blockers")
    if not native_g1_pass and not qualification_blockers:
        raise ValueError("unmet native gates must name at least one blocker")
