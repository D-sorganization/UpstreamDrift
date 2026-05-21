"""Pure consensus computation for peer-review verdicts (Tools #2738).

The consensus algorithm is intentionally pure (no I/O, no async, no
registry access) so it can be unit-tested exhaustively and reused by any
caller that already has a list of verdicts.

Algorithm
---------
1. ``abstain`` verdicts contribute zero weight.
2. Each non-abstain verdict contributes ``confidence_0_to_1`` to the
   weighted score for its verdict bucket. The role tie-breaker adds a
   tiny epsilon (specialist > critic > advocate) so that perfectly tied
   scores resolve deterministically rather than collapsing to
   ``no_consensus``.
3. The bucket with the highest weighted score wins:

   - ``approve``         → ``"approved"``
   - ``reject``          → ``"rejected"``
   - ``request_changes`` → ``"needs_revision"``

4. If all non-abstain weights are zero (i.e. all verdicts were abstain),
   or if the top two buckets tie exactly even after the role tie-breaker,
   we return ``"no_consensus"``.

The tie-breaker epsilon is small enough that any non-trivial confidence
delta dominates it; it only matters for verdicts that are otherwise
algebraically equal.
"""

from __future__ import annotations

from collections.abc import Sequence

from .contracts import ConsensusKind, ReviewerRole, ReviewVerdict

# Role weight bumps. Order encodes: specialist > critic > advocate.
# Values are tiny so they only break exact ties.
_ROLE_TIEBREAKER: dict[ReviewerRole, float] = {
    "specialist": 3e-6,
    "critic": 2e-6,
    "advocate": 1e-6,
}


# Verdict → consensus bucket mapping. ``abstain`` is intentionally absent.
_VERDICT_TO_CONSENSUS: dict[str, ConsensusKind] = {
    "approve": "approved",
    "reject": "rejected",
    "request_changes": "needs_revision",
}


def compute_consensus(verdicts: Sequence[ReviewVerdict]) -> ConsensusKind:
    """Compute the consensus disposition for a sequence of verdicts.

    Precondition: ``verdicts`` is non-empty. Empty input raises
    :class:`ValueError` because "no verdicts at all" is a programmer
    error, distinct from "all reviewers abstained" (which is a legitimate
    runtime outcome and returns ``"no_consensus"``).
    """
    if not verdicts:
        raise ValueError("compute_consensus requires at least one verdict")

    scores: dict[str, float] = {
        "approve": 0.0,
        "reject": 0.0,
        "request_changes": 0.0,
    }
    for v in verdicts:
        if v.verdict == "abstain":
            continue
        scores[v.verdict] += v.confidence_0_to_1
        scores[v.verdict] += _ROLE_TIEBREAKER[v.reviewer_role]

    total = sum(scores.values())
    if total == 0.0:
        return "no_consensus"

    # Find the top bucket. If two buckets tie exactly (including after
    # the tie-breaker), report no_consensus.
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_name, top_score = ranked[0]
    if len(ranked) > 1 and ranked[1][1] == top_score:
        return "no_consensus"

    return _VERDICT_TO_CONSENSUS[top_name]


__all__ = ["compute_consensus"]
