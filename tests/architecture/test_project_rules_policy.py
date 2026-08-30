from pathlib import Path

import pytest


PROJECT_RULES = (
    Path(__file__).resolve().parents[2]
    / ".gaai/project/contexts/rules/project.rules.md"
)


@pytest.mark.unit
def test_project_rules_match_authoritative_main_branch_policy() -> None:
    rules = PROJECT_RULES.read_text(encoding="utf-8")

    assert "PRs target `main`." in rules
    assert "automated merge may be enabled" in rules
    assert "Human approval required" not in rules
    assert "All AI work on `staging`" not in rules
