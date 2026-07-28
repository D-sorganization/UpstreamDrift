from __future__ import annotations

from pathlib import Path

import pytest

from scripts.check_agent_governance_consistency import audit_repository

pytestmark = pytest.mark.unit


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_policy_files(root: Path, *, project_rules: str, gaai_core: str) -> None:
    _write(
        root / "CLAUDE.md",
        "> PRs target `main`. Use focused topic branches such as `fix/...`.\n",
    )
    _write(
        root / ".gaai" / "project" / "contexts" / "rules" / "project.rules.md",
        project_rules,
    )
    _write(root / ".gaai" / "core" / "GAAI.md", gaai_core)


def test_audit_repository_accepts_main_based_governance(tmp_path: Path) -> None:
    _write_policy_files(
        tmp_path,
        project_rules=(
            "1. All AI work uses `main`-based topic branches. "
            "Never commit directly to `main`.\n"
            "2. PRs target `main`. Auto-merge may be enabled only after "
            "required checks pass.\n"
        ),
        gaai_core=(
            "AI agents work on **`main`-based topic branches**. "
            "Promotion happens by PR into `main`.\n"
        ),
    )
    _write(
        tmp_path / ".github" / "workflows" / "agent.yml",
        "on:\n  pull_request:\n    branches: [main]\n"
        'jobs:\n  branch:\n    steps:\n      - run: git checkout -B "$BRANCH_NAME" origin/main\n',
    )

    assert audit_repository(tmp_path) == []


def test_audit_repository_rejects_staging_policy_conflicts(tmp_path: Path) -> None:
    _write_policy_files(
        tmp_path,
        project_rules=(
            "1. All AI work on `staging` branch. Never commit directly to `main`.\n"
            "2. PRs target `staging`. No auto-merge. Human approval required.\n"
        ),
        gaai_core="AI agents work exclusively on the **`staging`** branch.\n",
    )

    findings = audit_repository(tmp_path)

    assert (
        ".gaai/project/contexts/rules/project.rules.md still directs AI work "
        "or PRs to staging"
    ) in findings
    assert (
        ".gaai/core/GAAI.md still documents staging as the AI work branch" in findings
    )


def test_audit_repository_rejects_automation_based_on_staging(
    tmp_path: Path,
) -> None:
    _write_policy_files(
        tmp_path,
        project_rules=(
            "1. All AI work uses `main`-based topic branches. "
            "Never commit directly to `main`.\n"
            "2. PRs target `main`. Auto-merge may be enabled only after "
            "required checks pass.\n"
        ),
        gaai_core="AI agents work on **`main`-based topic branches**.\n",
    )
    _write(
        tmp_path / ".github" / "workflows" / "agent.yml",
        "on:\n  pull_request:\n    branches: [staging]\n"
        'jobs:\n  branch:\n    steps:\n      - run: git checkout -B "$BRANCH_NAME" origin/staging\n',
    )

    findings = audit_repository(tmp_path)

    assert (
        ".github/workflows/agent.yml creates agent branches from origin/staging"
    ) in findings
    assert (
        ".github/workflows/agent.yml has a pull_request filter that omits main"
    ) in findings
