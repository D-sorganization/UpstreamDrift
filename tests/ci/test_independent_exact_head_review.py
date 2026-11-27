"""Hermetic tests for scripts/independent_exact_head_review.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "independent_exact_head_review.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "independent_exact_head_review", _MODULE_PATH
)
assert _SPEC and _SPEC.loader
mod = importlib.util.module_from_spec(_SPEC)
sys.modules["independent_exact_head_review"] = mod
_SPEC.loader.exec_module(mod)

pytestmark = pytest.mark.unit

HEAD = "c9b8b1448c588a9bb348cc7930895fbf9940bcdb"

_REVIEW = (
    "Independent Exact-Head Acceptance Review (d-sorgcodexagent / `codex`)\n\n"
    "Verdict: {verdict} @{sha}\n"
)


def _comment(
    body: str,
    *,
    created_at: str = "2026-11-22T18:00:00Z",
    login: str = "d-sorgcodexagent[bot]",
) -> dict[str, str]:
    """comment."""
    return {"body": body, "created_at": created_at, "user": {"login": login}}


def test_pass_at_exact_head_clears_hold() -> None:
    """test pass at exact head clears hold."""
    reviews = mod.parse_issue_comments(
        [_comment(_REVIEW.format(verdict="PASS", sha=HEAD))]
    )
    reasons = mod.independent_review_hold_reason(
        reviews,
        head_sha=HEAD,
        author_model="claude",
        labels=["agent-model:claude"],
    )
    assert reasons == []


def test_fix_needed_at_exact_head_holds() -> None:
    """test fix needed at exact head holds."""
    reviews = mod.parse_issue_comments(
        [_comment(_REVIEW.format(verdict="FIX-NEEDED", sha=HEAD))]
    )
    reasons = mod.independent_review_hold_reason(
        reviews,
        head_sha=HEAD,
        author_model="claude",
        labels=["agent-model:claude"],
    )
    assert len(reasons) == 1
    assert "FIX-NEEDED" in reasons[0]
    assert HEAD in reasons[0]


def test_arming_refused_without_pass_at_exact_head() -> None:
    """test arming refused without pass at exact head."""
    reasons = mod.independent_review_hold_reason(
        [],
        head_sha=HEAD,
        author_model="claude",
        labels=["agent-model:claude"],
    )
    assert reasons
    assert "no independent exact-head verdict" in reasons[0].lower()


def test_quoted_fix_needed_in_reply_does_not_trigger() -> None:
    """test quoted fix needed in reply does not trigger."""
    body = (
        "Independent Exact-Head Acceptance Review (d-sorgcodexagent / `codex`)\n"
        "Follow-up: the prior note said `FIX-NEEDED @"
        + HEAD
        + "` but the current verdict is PASS @"
        + HEAD
    )
    parsed = mod.parse_independent_review_comment(body)
    assert parsed is not None
    assert parsed.verdict == "PASS"


def test_untrusted_commenter_pass_is_ignored() -> None:
    """test untrusted commenter pass is ignored."""
    body = _REVIEW.format(verdict="PASS", sha=HEAD)
    reviews = mod.parse_issue_comments([_comment(body, login="evil-contributor")])
    assert reviews == []
    reasons = mod.independent_review_hold_reason(
        reviews,
        head_sha=HEAD,
        author_model="claude",
        labels=["agent-model:claude"],
    )
    assert reasons


def test_gate_fail_closed_without_agent_labels() -> None:
    """test gate fail closed without agent labels."""
    assert mod.independent_review_gate_active([], author_model="", labels=[]) is True
    reasons = mod.independent_review_hold_reason(
        [],
        head_sha=HEAD,
        author_model="claude",
        labels=[],
    )
    assert reasons
    assert "no independent exact-head verdict" in reasons[0].lower()


def test_exempt_label_disables_gate() -> None:
    """test exempt label disables gate."""
    assert (
        mod.independent_review_gate_active(
            [],
            author_model="claude",
            labels=[mod.INDEPENDENT_EXACT_HEAD_EXEMPT_LABEL],
        )
        is False
    )
    reasons = mod.independent_review_hold_reason(
        [],
        head_sha=HEAD,
        author_model="claude",
        labels=[mod.INDEPENDENT_EXACT_HEAD_EXEMPT_LABEL],
    )
    assert reasons == []


def test_stale_head_fix_needed_does_not_hold_current_head() -> None:
    """test stale head fix needed does not hold current head."""
    old = "a" * 40
    reviews = mod.parse_issue_comments(
        [_comment(_REVIEW.format(verdict="FIX-NEEDED", sha=old))]
    )
    reasons = mod.independent_review_hold_reason(
        reviews,
        head_sha=HEAD,
        author_model="claude",
        labels=["agent-model:claude"],
    )
    assert reasons
    assert "no independent exact-head verdict" in reasons[0].lower()


def test_spoofed_commit_trailer_is_not_an_author_model() -> None:
    """Commit messages are author-controlled and cannot name the author model."""
    messages = [
        "feat: real work\n\nAgent-Id: claude\nModel: claude\n",
    ]
    assert mod.author_model_from_commit_messages(messages) == ""
    reviews = mod.parse_issue_comments(
        [_comment(_REVIEW.format(verdict="PASS", sha=HEAD))]
    )
    reasons = mod.independent_review_hold_reason(
        reviews,
        head_sha=HEAD,
        author_model=mod.author_model_from_commit_messages(messages),
        labels=[],
    )
    assert reasons
    assert "author model unknown" in reasons[0].lower()
