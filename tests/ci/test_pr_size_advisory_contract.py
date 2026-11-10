"""CI contract: PR changed-line count is advisory, not a merge blocker (#14820)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "pr-size-gate.yml"
CHECKER = REPO_ROOT / "scripts" / "ci" / "check_pr_diff_size.py"
AGENTS_DOC = REPO_ROOT / "docs" / "agents" / "pr-size-advisory.md"


def test_pr_size_workflow_is_advisory_not_blocking() -> None:
    """Owner order 2026-11-06: the pr-size job must not pass --strict in CI."""
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "PR Size Advisory" in text
    assert "not a merge blocker" in text or "advisory" in text
    assert "--strict" not in text
    assert "check_pr_diff_size.py" in text
    assert "name: pr-size" in text


def test_pr_size_checker_default_is_non_blocking() -> None:
    """The checker documents advisory default exit 0; --strict is opt-in only."""
    text = CHECKER.read_text(encoding="utf-8")
    assert "RD #4118" in text or "#14741" in text
    assert "--strict" in text
    assert "exit 1 when over cap" in text or "exit 1 when over the cap" in text
    assert "exits **0**" in text or "Default exit code is **0**" in text


def test_pr_size_agent_doc_records_owner_order() -> None:
    text = AGENTS_DOC.read_text(encoding="utf-8")
    assert "2026-11-06" in text
    assert "Dieter" in text
    assert "#14820" in text
