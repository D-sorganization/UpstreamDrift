#!/usr/bin/env python3
"""Audit GitHub Actions workflow and agent-configuration inventory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORKFLOW_GLOB_PATTERNS = ("*.yml", "*.yaml")
# Bumped 2026-05-18 from 71 -> 80 to accommodate guard workflows
# (Jules-Diff-Verifier, Stub-Introduction-Guard, Verify-Issue-Closure,
# anti-phantom-merge, issue-closure-policy, lint-workflow-files) that
# were added during the fleet-wide CI hardening initiative. The
# 25-workflow consolidation target remains; see issue #3835.
DEFAULT_MAX_ACTIVE_WORKFLOWS = 80
AGENT_CONFIG_ROOTS = (".claude", ".gaai", ".agent", ".kiro", ".jules")
INVENTORY_PATH = Path(".github/WORKFLOWS.md")
AGENT_GOVERNANCE_PATH = Path("docs/development/agents/migration.md")
AGENT_BUDGET_CONTRACT_HEADING = "## Agent Automation Budget Contract"
AGENT_BUDGET_CONTRACT_TERMS = (
    "Max wall time",
    "Max parallel sessions",
    "Concurrency",
    "Audit artifact",
    "Human approval",
)


def iter_active_workflows(workflow_dir: Path) -> list[Path]:
    """Return active workflow YAML files directly under `.github/workflows`.

    Postcondition: archived workflows in nested directories are excluded.
    """
    if not workflow_dir.exists():
        raise FileNotFoundError(f"workflow directory does not exist: {workflow_dir}")

    return sorted(
        path
        for pattern in WORKFLOW_GLOB_PATTERNS
        for path in workflow_dir.glob(pattern)
        if path.is_file()
    )


def parse_inventory_table(path: Path) -> dict[str, list[str]]:
    """Return workflow inventory rows keyed by workflow file name.

    Postcondition: separator and header rows are ignored.
    """
    if not path.exists():
        raise FileNotFoundError(f"workflow inventory does not exist: {path}")

    rows: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) < 6:
            continue
        first_cell = cells[0]
        if first_cell in {"File", "------"} or set(first_cell) == {"-"}:
            continue
        rows[first_cell] = cells
    return rows


def parse_documented_agent_roots(path: Path) -> set[str]:
    """Return agent config roots documented in the migration inventory.

    Postcondition: returned directory names include a trailing slash.
    """
    if not path.exists():
        return set()

    roots: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if not cells:
            continue
        directory = cells[0]
        if directory in {"Directory", "-----------"}:
            continue
        if directory.startswith("."):
            roots.add(_normalize_agent_root(directory))
    return roots


def audit_workflow_inventory(
    repo_root: Path,
    max_active_workflows: int,
) -> list[str]:
    """Audit active workflow files against `.github/WORKFLOWS.md`.

    Postcondition: every active workflow has exactly one inventory row, every
    row points at an active workflow, and the active count does not exceed cap.
    """
    if max_active_workflows < 1:
        raise ValueError("max_active_workflows must be greater than zero")

    workflow_dir = repo_root / ".github" / "workflows"
    workflow_files = iter_active_workflows(workflow_dir)
    workflow_names = {path.name for path in workflow_files}
    findings: list[str] = []

    if len(workflow_files) > max_active_workflows:
        findings.append(
            f"active workflow count {len(workflow_files)} exceeds cap "
            f"{max_active_workflows}"
        )

    try:
        inventory_rows = parse_inventory_table(repo_root / INVENTORY_PATH)
    except FileNotFoundError as exc:
        return [str(exc)]

    for workflow_name in sorted(workflow_names):
        if workflow_name not in inventory_rows:
            findings.append(
                f"workflow missing from {INVENTORY_PATH.as_posix()}: {workflow_name}"
            )

    for workflow_name in sorted(inventory_rows):
        if workflow_name not in workflow_names:
            findings.append(
                f"inventory row references missing workflow: {workflow_name}"
            )

    return findings


def audit_permissions(repo_root: Path) -> list[str]:
    """Audit active workflow files for unsafe top-level permissions settings."""
    findings: list[str] = []
    workflow_dir = repo_root / ".github" / "workflows"
    for workflow_file in iter_active_workflows(workflow_dir):
        rel_path = workflow_file.relative_to(repo_root).as_posix()
        lines = workflow_file.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, 1):
            if not line.startswith("permissions:"):
                continue
            value = line.split(":", 1)[1].strip()
            if value == "write-all":
                findings.append(f"{rel_path}:{line_number} uses permissions: write-all")
            if value == "" and _permissions_block_is_empty(lines, line_number):
                findings.append(
                    f"{rel_path}:{line_number} has an unconstrained permissions block"
                )
    return findings


def audit_agent_config_roots(repo_root: Path) -> list[str]:
    """Audit root-level agent config directories against migration ownership docs."""
    documented_roots = parse_documented_agent_roots(repo_root / AGENT_GOVERNANCE_PATH)
    findings: list[str] = []

    for root_name in AGENT_CONFIG_ROOTS:
        root_path = repo_root / root_name
        if not root_path.exists():
            continue
        normalized = _normalize_agent_root(root_name)
        if normalized not in documented_roots:
            findings.append(
                "agent config root exists without "
                f"{AGENT_GOVERNANCE_PATH.as_posix()} row: {normalized}"
            )

    return findings


def audit_agent_budget_contract(repo_root: Path) -> list[str]:
    """Audit that mutating agent workflows have documented safety budgets."""
    inventory_path = repo_root / INVENTORY_PATH
    try:
        inventory_rows = parse_inventory_table(inventory_path)
    except FileNotFoundError as exc:
        return [str(exc)]

    mutating_agent_workflows = [
        file_name
        for file_name, cells in inventory_rows.items()
        if len(cells) >= 4 and cells[2] == "@agents" and "write" in cells[3]
    ]
    if not mutating_agent_workflows:
        return []

    inventory_text = inventory_path.read_text(encoding="utf-8")
    missing_terms = [
        term for term in AGENT_BUDGET_CONTRACT_TERMS if term not in inventory_text
    ]
    if AGENT_BUDGET_CONTRACT_HEADING not in inventory_text or missing_terms:
        return [
            f"{INVENTORY_PATH.as_posix()} is missing Agent Automation Budget "
            "Contract for mutating agent workflows"
        ]
    return []


def audit_repository(
    repo_root: Path,
    max_active_workflows: int = DEFAULT_MAX_ACTIVE_WORKFLOWS,
) -> list[str]:
    """Audit workflow and agent ownership inventory for the repository."""
    if not repo_root.exists():
        raise FileNotFoundError(f"repository root does not exist: {repo_root}")
    if not repo_root.is_dir():
        raise NotADirectoryError(f"repository root is not a directory: {repo_root}")

    findings: list[str] = []
    findings.extend(audit_workflow_inventory(repo_root, max_active_workflows))
    findings.extend(audit_permissions(repo_root))
    findings.extend(audit_agent_config_roots(repo_root))
    findings.extend(audit_agent_budget_contract(repo_root))
    return findings


def _permissions_block_is_empty(lines: list[str], line_number: int) -> bool:
    start_index = line_number
    for line in lines[start_index:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        return not line.startswith(" ")
    return True


def _normalize_agent_root(root_name: str) -> str:
    trimmed = root_name.strip()
    return trimmed if trimmed.endswith("/") else f"{trimmed}/"


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser."""
    parser = argparse.ArgumentParser(
        description="Fail when workflow or agent inventory drifts."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root to audit.",
    )
    parser.add_argument(
        "--max-active-workflows",
        type=int,
        default=DEFAULT_MAX_ACTIVE_WORKFLOWS,
        help=(
            "Maximum active workflow YAML files allowed directly under "
            ".github/workflows."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the workflow inventory audit."""
    args = build_parser().parse_args(argv)
    findings = audit_repository(
        args.repo_root.resolve(),
        max_active_workflows=args.max_active_workflows,
    )
    if findings:
        for finding in findings:
            print(finding, file=sys.stderr)
        return 1

    print("Workflow and agent inventory is documented and within guardrails.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
