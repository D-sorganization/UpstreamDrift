from __future__ import annotations

from pathlib import Path

from scripts.check_workflow_inventory import audit_repository


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _workflow(content: str = "permissions: contents: read\n") -> str:
    return f"name: Test\non: workflow_dispatch\n{content}\njobs: {{}}\n"


def _inventory_row(file_name: str, owner: str = "@core") -> str:
    return (
        f"| {file_name} | workflow_dispatch | {owner} | contents: read | "
        "KEEP: Test workflow | n/a |\n"
    )


def _write_inventory(root: Path, rows: str) -> None:
    _write(
        root / ".github" / "WORKFLOWS.md",
        "# Workflow Inventory\n\n"
        "| File | Trigger | Owner | Permissions | Purpose | Replaceable by |\n"
        "|------|---------|-------|-------------|---------|----------------|\n"
        f"{rows}",
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


def test_audit_repository_rejects_growth_above_configured_cap(tmp_path: Path) -> None:
    _write(tmp_path / ".github" / "workflows" / "ci.yml", _workflow())
    _write(tmp_path / ".github" / "workflows" / "extra.yml", _workflow())
    _write_inventory(tmp_path, _inventory_row("ci.yml") + _inventory_row("extra.yml"))
    _write_agent_governance(tmp_path, "")

    findings = audit_repository(tmp_path, max_active_workflows=1)

    assert "active workflow count 2 exceeds cap 1" in findings


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


def test_checked_in_repository_inventory_is_current() -> None:
    """The default guard must pass on the checked-in repository state."""
    assert audit_repository(Path(".")) == []
