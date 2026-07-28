#!/usr/bin/env python3
"""Fail when repo-local agent governance disagrees on the branch model."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLAUDE = Path("CLAUDE.md")
GAAI_PROJECT_RULES = Path(".gaai/project/contexts/rules/project.rules.md")
GAAI_CORE = Path(".gaai/core/GAAI.md")
WORKFLOW_DIR = Path(".github/workflows")
CANONICAL_PR_TARGET = "main"
STAGING_POLICY_MARKERS = (
    "All AI work on `staging` branch",
    "PRs target `staging`",
)
STAGING_CORE_MARKERS = (
    "work exclusively on the **`staging`** branch",
    "staging  \u2190\u2500\u2500 AI works here",
    "in_progress` on staging",
)
ORIGIN_STAGING = re.compile(r"\borigin/staging\b")
PULL_REQUEST_STAGING_ONLY = re.compile(
    r"pull_request:\s*(?:(?!\n\S).)*?branches:\s*\[\s*staging\s*\]",
    re.DOTALL,
)


def audit_repository(repo_root: Path) -> list[str]:
    """Audit branch-model policy files and agent workflow branch bases."""
    if not repo_root.exists():
        raise FileNotFoundError(f"repository root does not exist: {repo_root}")
    if not repo_root.is_dir():
        raise NotADirectoryError(f"repository root is not a directory: {repo_root}")

    findings: list[str] = []
    findings.extend(_audit_policy_files(repo_root))
    findings.extend(_audit_workflows(repo_root))
    return findings


def _read_required(repo_root: Path, relative_path: Path) -> str:
    path = repo_root / relative_path
    if not path.exists():
        raise FileNotFoundError(f"required governance file is missing: {relative_path}")
    return path.read_text(encoding="utf-8")


def _audit_policy_files(repo_root: Path) -> list[str]:
    findings: list[str] = []
    claude = _read_required(repo_root, CLAUDE)
    project_rules = _read_required(repo_root, GAAI_PROJECT_RULES)
    gaai_core = _read_required(repo_root, GAAI_CORE)

    if "PRs target `main`" not in claude:
        findings.append("CLAUDE.md must declare PRs target `main`")
    if "PRs target `main`" not in project_rules:
        findings.append(
            ".gaai/project/contexts/rules/project.rules.md must declare PRs "
            "target `main`"
        )
    if any(marker in project_rules for marker in STAGING_POLICY_MARKERS):
        findings.append(
            ".gaai/project/contexts/rules/project.rules.md still directs AI work "
            "or PRs to staging"
        )
    if any(marker in gaai_core for marker in STAGING_CORE_MARKERS):
        findings.append(
            ".gaai/core/GAAI.md still documents staging as the AI work branch"
        )
    if "`main`-based topic branches" not in project_rules:
        findings.append(
            ".gaai/project/contexts/rules/project.rules.md must require "
            "`main`-based topic branches"
        )
    if "`main`-based topic branches" not in gaai_core:
        findings.append(".gaai/core/GAAI.md must document `main`-based topic branches")

    return findings


def _audit_workflows(repo_root: Path) -> list[str]:
    workflow_dir = repo_root / WORKFLOW_DIR
    if not workflow_dir.exists():
        return []

    findings: list[str] = []
    for workflow_file in sorted(workflow_dir.glob("*.yml")) + sorted(
        workflow_dir.glob("*.yaml")
    ):
        text = workflow_file.read_text(encoding="utf-8")
        relative_path = workflow_file.relative_to(repo_root).as_posix()
        if ORIGIN_STAGING.search(text):
            findings.append(
                f"{relative_path} creates agent branches from origin/staging"
            )
        if PULL_REQUEST_STAGING_ONLY.search(text):
            findings.append(
                f"{relative_path} has a pull_request filter that omits main"
            )
    return findings


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Fail when agent governance branch policy drifts."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=ROOT,
        help="Repository root to audit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the branch-model governance consistency audit."""
    args = build_parser().parse_args(argv)
    findings = audit_repository(args.repo_root.resolve())
    if findings:
        for finding in findings:
            print(finding, file=sys.stderr)
        return 1
    print(f"Agent governance consistently targets {CANONICAL_PR_TARGET}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
