from __future__ import annotations

from pathlib import Path

import pytest

import scripts.check_workflow_inventory as workflow_inventory
from scripts.check_workflow_inventory import audit_repository


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _workflow(
    content: str = "permissions: contents: read\n",
    *,
    trigger: str = "workflow_dispatch",
) -> str:
    return f"name: Test\non: {trigger}\n{content}\njobs: {{}}\n"


def _inventory_row(file_name: str, owner: str = "@core") -> str:
    return (
        f"| {file_name} | workflow_dispatch | {owner} | contents: read | "
        "KEEP: Test workflow | n/a |\n"
    )


def _workflow_budget_contract(
    *,
    max_active_workflows: int,
    max_pr_triggered_workflows: int,
    consolidation_target: int = 25,
) -> str:
    return (
        f"The current durable guardrail is a no-growth cap at "
        f"{max_active_workflows} active workflows and "
        f"{max_pr_triggered_workflows} PR-triggered workflows. The "
        f"consolidation target for issue #3835 remains "
        f"{consolidation_target} active workflows or fewer after owners "
        "validate low-risk removals.\n\n"
    )


def _write_inventory(
    root: Path,
    rows: str,
    *,
    max_active_workflows: int = 83,
    max_pr_triggered_workflows: int = 33,
) -> None:
    _write(
        root / ".github" / "WORKFLOWS.md",
        "# Workflow Inventory\n\n"
        + _workflow_budget_contract(
            max_active_workflows=max_active_workflows,
            max_pr_triggered_workflows=max_pr_triggered_workflows,
        )
        + "| File | Trigger | Owner | Permissions | Purpose | Replaceable by |\n"
        "|------|---------|-------|-------------|---------|----------------|\n"
        f"{rows}",
    )


def _agent_budget_contract() -> str:
    return (
        "\n## Agent Automation Budget Contract\n\n"
        "- Max wall time: every mutating agent workflow must declare a timeout.\n"
        "- Max parallel sessions: workflows that launch agents must bound parallelism.\n"
        "- Concurrency: mutating workflows must define a concurrency/idempotency plan.\n"
        "- Audit artifact: prompts, refs, modified files, and outcomes must be durable.\n"
        "- Human approval: destructive or release-impacting actions require owner review.\n"
    )


def _write_agent_governance(root: Path, rows: str) -> None:
    _write(
        root / "docs" / "development" / "agents" / "migration.md",
        "# Agent Configuration Ownership\n\n"
        "| Directory | Status | Owner | Migration note |\n"
        "|-----------|--------|-------|----------------|\n"
        f"{rows}",
    )


def test_audit_repository_accepts_documented_workflows_and_agent_roots(
    tmp_path: Path,
) -> None:
    _write(tmp_path / ".github" / "workflows" / "ci.yml", _workflow())
    _write_inventory(tmp_path, _inventory_row("ci.yml"))
    _write(tmp_path / ".gaai" / "README.md", "# canonical\n")
    _write_agent_governance(
        tmp_path,
        "| .gaai/ | Canonical | @core | Authoritative GAAI root. |\n",
    )

    assert audit_repository(tmp_path, max_active_workflows=1) == []


def test_audit_repository_rejects_workflows_missing_inventory_rows(
    tmp_path: Path,
) -> None:
    _write(tmp_path / ".github" / "workflows" / "ci.yml", _workflow())
    _write_inventory(tmp_path, "")
    _write_agent_governance(tmp_path, "")

    findings = audit_repository(tmp_path, max_active_workflows=1)

    assert "workflow missing from .github/WORKFLOWS.md: ci.yml" in findings


def test_audit_repository_rejects_inventory_rows_for_missing_workflows(
    tmp_path: Path,
) -> None:
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    _write_inventory(tmp_path, _inventory_row("missing.yml"))
    _write_agent_governance(tmp_path, "")

    findings = audit_repository(tmp_path, max_active_workflows=1)

    assert "inventory row references missing workflow: missing.yml" in findings


def test_audit_repository_rejects_duplicate_inventory_rows(tmp_path: Path) -> None:
    _write(tmp_path / ".github" / "workflows" / "ci.yml", _workflow())
    _write_inventory(
        tmp_path,
        _inventory_row("ci.yml") + _inventory_row("ci.yml", owner="@infra"),
    )
    _write_agent_governance(tmp_path, "")

    findings = audit_repository(tmp_path)

    assert (
        ".github/WORKFLOWS.md has duplicate workflow row: ci.yml (2 rows)" in findings
    )


def test_audit_repository_rejects_duplicate_rows_for_missing_workflows(
    tmp_path: Path,
) -> None:
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    _write_inventory(
        tmp_path,
        _inventory_row("missing.yml") + _inventory_row("missing.yml", owner="@infra"),
    )
    _write_agent_governance(tmp_path, "")

    findings = audit_repository(tmp_path)

    assert (
        ".github/WORKFLOWS.md has duplicate workflow row: missing.yml (2 rows)"
    ) in findings
    assert "inventory row references missing workflow: missing.yml" in findings


def test_audit_repository_rejects_growth_above_configured_cap(tmp_path: Path) -> None:
    _write(tmp_path / ".github" / "workflows" / "ci.yml", _workflow())
    _write(tmp_path / ".github" / "workflows" / "extra.yml", _workflow())
    _write_inventory(
        tmp_path,
        _inventory_row("ci.yml") + _inventory_row("extra.yml"),
        max_active_workflows=2,
    )
    _write_agent_governance(tmp_path, "")

    findings = audit_repository(tmp_path, max_active_workflows=1)

    assert "active workflow count 2 exceeds cap 1" in findings


def test_audit_repository_rejects_pr_triggered_growth_above_configured_cap(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / ".github" / "workflows" / "ci.yml",
        _workflow(trigger="pull_request"),
    )
    _write_inventory(
        tmp_path,
        _inventory_row("ci.yml").replace("workflow_dispatch", "pull_request"),
        max_active_workflows=1,
        max_pr_triggered_workflows=1,
    )
    _write_agent_governance(tmp_path, "")

    findings = audit_repository(
        tmp_path,
        max_active_workflows=1,
        max_pr_triggered_workflows=0,
    )

    assert "PR-triggered workflow count 1 exceeds cap 0" in findings


def test_audit_repository_rejects_documented_budget_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write(tmp_path / ".github" / "workflows" / "ci.yml", _workflow())
    _write_inventory(
        tmp_path,
        _inventory_row("ci.yml"),
        max_active_workflows=1,
    )
    _write_agent_governance(tmp_path, "")
    monkeypatch.setattr(workflow_inventory, "DEFAULT_MAX_ACTIVE_WORKFLOWS", 2)

    findings = audit_repository(tmp_path)

    assert (
        ".github/WORKFLOWS.md documents max_active_workflows 1 but "
        "scripts/check_workflow_inventory.py defaults to 2"
    ) in findings


def test_audit_repository_rejects_documented_pr_budget_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write(
        tmp_path / ".github" / "workflows" / "ci.yml",
        _workflow(trigger="pull_request"),
    )
    _write_inventory(
        tmp_path,
        _inventory_row("ci.yml").replace("workflow_dispatch", "pull_request"),
        max_pr_triggered_workflows=1,
    )
    _write_agent_governance(tmp_path, "")
    monkeypatch.setattr(workflow_inventory, "DEFAULT_MAX_PR_TRIGGERED_WORKFLOWS", 2)

    findings = audit_repository(tmp_path)

    assert (
        ".github/WORKFLOWS.md documents max_pr_triggered_workflows 1 but "
        "scripts/check_workflow_inventory.py defaults to 2"
    ) in findings


def test_audit_repository_rejects_write_all_permissions(tmp_path: Path) -> None:
    _write(
        tmp_path / ".github" / "workflows" / "ci.yml",
        _workflow("permissions: write-all\n"),
    )
    _write_inventory(tmp_path, _inventory_row("ci.yml"))
    _write_agent_governance(tmp_path, "")

    findings = audit_repository(tmp_path, max_active_workflows=1)

    assert ".github/workflows/ci.yml:3 uses permissions: write-all" in findings


def test_audit_repository_rejects_undocumented_agent_roots(tmp_path: Path) -> None:
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    _write_inventory(tmp_path, "")
    _write(tmp_path / ".gaai" / "README.md", "# canonical\n")
    _write(tmp_path / ".claude" / "README.md", "# mirror\n")
    _write_agent_governance(
        tmp_path,
        "| .gaai/ | Canonical | @core | Authoritative GAAI root. |\n",
    )

    findings = audit_repository(tmp_path, max_active_workflows=1)

    assert (
        "agent config root exists without docs/development/agents/migration.md row: "
        ".claude/"
    ) in findings


def test_audit_repository_rejects_mutating_agent_workflows_without_budget_contract(
    tmp_path: Path,
) -> None:
    _write(tmp_path / ".github" / "workflows" / "agent.yml", _workflow())
    _write_inventory(
        tmp_path,
        "| agent.yml | workflow_dispatch | @agents | contents: write | "
        "KEEP: mutating agent workflow. | n/a |\n",
    )
    _write_agent_governance(tmp_path, "")

    findings = audit_repository(tmp_path, max_active_workflows=1)

    assert (
        ".github/WORKFLOWS.md is missing Agent Automation Budget Contract "
        "for mutating agent workflows"
    ) in findings


def test_audit_repository_accepts_mutating_agent_workflows_with_budget_contract(
    tmp_path: Path,
) -> None:
    _write(tmp_path / ".github" / "workflows" / "agent.yml", _workflow())
    _write_inventory(
        tmp_path,
        "| agent.yml | workflow_dispatch | @agents | contents: write | "
        "KEEP: mutating agent workflow. | n/a |\n" + _agent_budget_contract(),
    )
    _write_agent_governance(tmp_path, "")

    assert audit_repository(tmp_path, max_active_workflows=1) == []


def test_checked_in_repository_inventory_is_current() -> None:
    """The default guard must pass on the checked-in repository state."""
    assert audit_repository(Path(".")) == []
