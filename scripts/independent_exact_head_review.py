"""Independent exact-head PR acceptance reviews (FIX-NEEDED / PASS at @sha)."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

INDEPENDENT_EXACT_HEAD_EXEMPT_LABEL = "independent-exact-head-exempt"
INDEPENDENT_REVIEW_MARKER = "Independent Exact-Head Acceptance Review"

_VERDICT_AT_SHA = re.compile(
    r"(?P<verdict>FIX-NEEDED|PASS)\s*@(?P<sha>[0-9a-fA-F]{6,40})",
    re.IGNORECASE,
)
_REVIEWER_MODEL = re.compile(
    rf"{re.escape(INDEPENDENT_REVIEW_MARKER)}\s*\([^)]*/\s*`(?P<model>[^`]+)`",
    re.IGNORECASE,
)
_AGENT_MODEL_LABEL = re.compile(r"^agent-model:(.+)$", re.IGNORECASE)
_AGENT_LABEL = re.compile(r"^agent:(.+)$", re.IGNORECASE)
_FENCED_CODE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`]*`")


@dataclass(frozen=True)
class IndependentHeadReview:
    verdict: str
    head_sha: str
    reviewer_model: str
    created_at: str


def commenter_login_from_item(item: Mapping[str, Any]) -> str:
    """Return the GitHub ``user.login`` for an issue-comment payload."""
    user = item.get("user")
    if isinstance(user, Mapping):
        return str(user.get("login") or "").strip()
    return ""


def is_trusted_independent_reviewer_login(login: str) -> bool:
    """Only fleet agent bot identities may post binding independent verdicts."""
    lowered = login.strip().lower()
    if not lowered:
        return False
    if not lowered.endswith("[bot]"):
        return False
    return lowered.startswith("d-sorg") and "agent" in lowered


def normalize_model_token(value: str) -> str:
    """Return a lowercased model token, or ``""`` when blank.

    Precondition: ``value`` is a ``str``.
    Postcondition: the result has no leading or trailing whitespace.
    """
    return value.strip().lower()


def head_sha_matches(recorded: str, head: str) -> bool:
    """True when ``recorded`` names the same commit as ``head`` (prefix-safe)."""
    rec, act = normalize_model_token(recorded), normalize_model_token(head)
    if not rec or not act:
        return False
    if rec == act:
        return True
    shorter, longer = (rec, act) if len(rec) <= len(act) else (act, rec)
    return len(shorter) >= 7 and longer.startswith(shorter)


def models_are_independent(author_model: str, reviewer_model: str) -> bool:
    """Independence is by model string (#4249); same model is never accepted."""
    author = normalize_model_token(author_model)
    reviewer = normalize_model_token(reviewer_model)
    if not author or not reviewer:
        return False
    return author != reviewer


def author_model_from_labels(labels: Sequence[str]) -> str:
    """author model from labels."""
    for label in labels:
        lowered = label.strip().lower()
        match = _AGENT_MODEL_LABEL.match(lowered)
        if match:
            return match.group(1).strip()
        match = _AGENT_LABEL.match(lowered)
        if match:
            return match.group(1).strip()
    return ""


def author_model_from_commit_messages(messages: Sequence[str]) -> str:
    """Ignore commit trailers. They are author-controlled, not authenticated.

    Precondition: ``messages`` is a sequence of commit message strings.
    Postcondition: always ``""``. Callers fail closed unless a protected
    ``agent-model:`` or ``agent:`` label names the author model.
    """
    del messages
    return ""


def _body_for_verdict_scan(body: str) -> str:
    """Drop quoted or cited verdict text so only live verdict lines match."""
    text = _FENCED_CODE.sub("", body)
    text = _INLINE_CODE.sub("", text)
    lines = [line for line in text.splitlines() if not line.lstrip().startswith(">")]
    return "\n".join(lines)


def parse_independent_review_comment(
    body: str, *, created_at: str = ""
) -> IndependentHeadReview | None:
    """Parse one issue comment body into a structured independent review."""
    if INDEPENDENT_REVIEW_MARKER not in body:
        return None
    model_match = _REVIEWER_MODEL.search(body)
    reviewer_model = model_match.group("model").strip() if model_match else ""
    scan = _body_for_verdict_scan(body)
    verdict_match = None
    for match in _VERDICT_AT_SHA.finditer(scan):
        verdict_match = match
    if verdict_match is None:
        return None
    return IndependentHeadReview(
        verdict=verdict_match.group("verdict").upper(),
        head_sha=verdict_match.group("sha"),
        reviewer_model=reviewer_model,
        created_at=created_at,
    )


def parse_issue_comments(
    comments: Sequence[Mapping[str, Any]],
) -> list[IndependentHeadReview]:
    """parse issue comments."""
    rows: list[IndependentHeadReview] = []
    for item in comments:
        if not is_trusted_independent_reviewer_login(commenter_login_from_item(item)):
            continue
        body = str(item.get("body") or "")
        parsed = parse_independent_review_comment(
            body, created_at=str(item.get("created_at") or "")
        )
        if parsed is not None:
            rows.append(parsed)
    return rows


def independent_review_gate_active(
    reviews: Sequence[IndependentHeadReview],
    *,
    author_model: str,
    labels: Sequence[str] = (),
) -> bool:
    """True when an exact-head independent verdict is required before arm or merge."""
    lowered = {label.strip().lower() for label in labels}
    if INDEPENDENT_EXACT_HEAD_EXEMPT_LABEL in lowered:
        return False
    _ = author_model  # retained for API stability; gate is fail-closed by default
    return True


def latest_independent_verdict_at_head(
    reviews: Sequence[IndependentHeadReview],
    *,
    head_sha: str,
    author_model: str,
) -> tuple[str | None, str]:
    """Return the latest eligible verdict at ``head_sha``, or (None, detail)."""
    eligible = [
        row
        for row in reviews
        if models_are_independent(author_model, row.reviewer_model)
        and head_sha_matches(row.head_sha, head_sha)
    ]
    if not eligible:
        return None, "no_independent_verdict_at_head"
    latest = max(eligible, key=lambda row: row.created_at or "")
    return latest.verdict, latest.verdict


def independent_review_hold_reason(
    reviews: Sequence[IndependentHeadReview],
    *,
    head_sha: str,
    author_model: str,
    labels: Sequence[str] = (),
    unreadable: bool = False,
) -> list[str]:
    """Signal 5: independent exact-head verdicts at the current head."""
    if unreadable:
        return ["independent exact-head review state is unreadable"]
    if not head_sha:
        return [
            "PR head SHA is unknown; independent exact-head review cannot be verified"
        ]
    if not independent_review_gate_active(
        reviews, author_model=author_model, labels=labels
    ):
        return []
    if not author_model:
        fix_at_head = [
            row
            for row in reviews
            if row.verdict == "FIX-NEEDED"
            and row.reviewer_model
            and head_sha_matches(row.head_sha, head_sha)
        ]
        if fix_at_head:
            return [f"independent exact-head review is FIX-NEEDED at {head_sha}"]
        return [
            "independent exact-head verdict at current head cannot be verified "
            "(author model unknown)"
        ]
    verdict, detail = latest_independent_verdict_at_head(
        reviews, head_sha=head_sha, author_model=author_model
    )
    if verdict == "FIX-NEEDED":
        return [f"independent exact-head review is FIX-NEEDED at {head_sha}"]
    if verdict == "PASS":
        return []
    return [f"no independent exact-head verdict at current head ({detail})"]
