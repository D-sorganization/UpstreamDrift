#!/usr/bin/env python3
"""Govern Jules push/PR noise workflow retirement batches (UD-CI-003 / #13575)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT / "scripts" / "config" / "jules_push_pr_noise_retirement.json"
)
ACTIVE_WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"
REQUIRED_CHECKS_PATH = REPO_ROOT / "scripts" / "config" / "ci-required-checks-live.json"


class RetirementConfigError(ValueError):
    """Raised when the retirement JSON contract is violated."""


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RetirementConfigError(f"missing config: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RetirementConfigError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise RetirementConfigError(f"{path} must be a JSON object")
    return data


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load and validate the retirement config.

    Postcondition: returned mapping includes ``k1_k2_keep`` and ``batches``.
    """
    data = _load_json(path)
    keep = data.get("k1_k2_keep")
    batches = data.get("batches")
    if not isinstance(keep, list) or not all(isinstance(x, str) and x for x in keep):
        raise RetirementConfigError("'k1_k2_keep' must be a non-empty string list")
    if not isinstance(batches, dict) or not batches:
        raise RetirementConfigError("'batches' must be a non-empty object")
    for batch_id, batch in batches.items():
        _validate_batch(batch_id, batch)
    return data


def _validate_batch(batch_id: str, batch: object) -> None:
    if not isinstance(batch, dict):
        raise RetirementConfigError(f"batch {batch_id} must be an object")
    retired = batch.get("retired_active_paths")
    archive_dir = batch.get("archive_dir")
    names = batch.get("workflow_names")
    if not isinstance(retired, list) or not retired:
        raise RetirementConfigError(f"batch {batch_id}: retired_active_paths required")
    if not all(
        isinstance(p, str) and p.startswith(".github/workflows/") for p in retired
    ):
        raise RetirementConfigError(
            f"batch {batch_id}: retired_active_paths must be under .github/workflows/"
        )
    if not isinstance(archive_dir, str) or not archive_dir.startswith(".github/"):
        raise RetirementConfigError(
            f"batch {batch_id}: archive_dir must be under .github/"
        )
    if not isinstance(names, list) or not all(isinstance(n, str) and n for n in names):
        raise RetirementConfigError(
            f"batch {batch_id}: workflow_names must be a string list"
        )
    if len(names) != len(retired):
        raise RetirementConfigError(
            f"batch {batch_id}: workflow_names and retired_active_paths must align"
        )


def all_retired_active_paths(config: dict[str, Any]) -> frozenset[str]:
    paths: set[str] = set()
    for batch in config["batches"].values():
        paths.update(batch["retired_active_paths"])
    return frozenset(paths)


def all_archive_basenames(config: dict[str, Any]) -> dict[str, str]:
    """Map active workflow basename to archive relative path."""
    mapping: dict[str, str] = {}
    for batch in config["batches"].values():
        archive = Path(batch["archive_dir"])
        for active_path in batch["retired_active_paths"]:
            basename = Path(active_path).name
            mapping[basename] = str(archive / basename)
    return mapping


def check_repository_state(repo_root: Path, config: dict[str, Any]) -> list[str]:
    """Return findings when retired paths reappear or archives are missing."""
    findings: list[str] = []
    keep_set = set(config["k1_k2_keep"])
    for batch in config["batches"].values():
        archive_root = repo_root / batch["archive_dir"]
        for active_path in batch["retired_active_paths"]:
            basename = Path(active_path).name
            active = repo_root / active_path
            archived = archive_root / basename
            if active.is_file():
                findings.append(f"retired workflow still active: {active_path}")
            if not archived.is_file():
                findings.append(
                    f"missing rollback archive: {archived.relative_to(repo_root)}"
                )
    workflows_dir = repo_root / ".github" / "workflows"
    if workflows_dir.is_dir():
        for path in sorted(workflows_dir.glob("Jules-*.yml")):
            if path.name not in keep_set:
                findings.append(
                    f"unexpected Jules workflow in active dir (not K1/K2): {path.name}"
                )
    for name in keep_set:
        if not (workflows_dir / name).is_file():
            findings.append(f"K1/K2 keep-set workflow missing: {name}")
    return findings


def check_pr_deletions(
    repo_root: Path,
    config: dict[str, Any],
    base_ref: str,
    head_ref: str = "HEAD",
) -> list[str]:
    """Ensure deleted workflow paths match the retirement allowlist exactly."""
    allowlist = all_retired_active_paths(config)
    deleted = _git_deleted_paths(
        repo_root, base_ref, head_ref, prefix=".github/workflows/"
    )
    unexpected = sorted(p for p in deleted if p not in allowlist)
    if unexpected:
        return [
            f"workflow deletion not on retirement allowlist: {p}" for p in unexpected
        ]
    return []


def _git_deleted_paths(
    repo_root: Path, base_ref: str, head_ref: str, prefix: str
) -> set[str]:
    cmd = [
        "git",
        "diff",
        "--name-only",
        "--diff-filter=D",
        f"{base_ref}...{head_ref}",
        "--",
        prefix,
    ]
    result = subprocess.run(
        cmd,
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git diff failed ({result.returncode}): {result.stderr.strip()}"
        )
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def required_contexts(config: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for batch in config["batches"].values():
        names.extend(batch["workflow_names"])
    return names


def check_not_required_contexts(config: dict[str, Any]) -> list[str]:
    if not REQUIRED_CHECKS_PATH.is_file():
        return []
    payload = json.loads(REQUIRED_CHECKS_PATH.read_text(encoding="utf-8"))
    required = payload.get("required_contexts", payload.get("contexts", []))
    if not isinstance(required, list):
        return ["ci-required-checks-live.json: required_contexts must be a list"]
    retired_names = set(required_contexts(config))
    hits = sorted(name for name in retired_names if name in required)
    if hits:
        return [
            f"retired workflow is still a required context: {name}" for name in hits
        ]
    return []


def disable_retired_workflows(
    repo: str,
    config: dict[str, Any],
    *,
    dry_run: bool = False,
) -> list[str]:
    """Disable GitHub workflows for each retired basename (post-merge hygiene)."""
    actions: list[str] = []
    for batch in config["batches"].values():
        for active_path in batch["retired_active_paths"]:
            workflow_ref = active_path.removeprefix(".github/workflows/")
            cmd = ["gh", "workflow", "disable", workflow_ref, "-R", repo]
            if dry_run:
                actions.append(f"would run: {' '.join(cmd)}")
                continue
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                actions.append(f"disabled: {workflow_ref}")
            elif "already disabled" in (result.stderr + result.stdout).lower():
                actions.append(f"already disabled: {workflow_ref}")
            else:
                actions.append(
                    f"disable failed for {workflow_ref}: "
                    f"{(result.stderr or result.stdout).strip()}"
                )
    return actions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Retirement config JSON path",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root (default: parent of scripts/)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate repository state and required-context contract",
    )
    parser.add_argument(
        "--check-pr-deletions",
        metavar="BASE_REF",
        help="Verify workflow deletions vs merge-base are allowlisted",
    )
    parser.add_argument(
        "--head-ref",
        default="HEAD",
        help="Head ref for --check-pr-deletions (default: HEAD)",
    )
    parser.add_argument(
        "--disable",
        action="store_true",
        help="Disable retired workflows via gh (needs actions:write token)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --disable, print commands without executing",
    )
    parser.add_argument(
        "--repo",
        default="",
        help="owner/repo for --disable (default: GITHUB_REPOSITORY)",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    findings: list[str] = []
    if args.check:
        findings.extend(check_repository_state(args.repo_root, config))
        findings.extend(check_not_required_contexts(config))
    if args.check_pr_deletions:
        findings.extend(
            check_pr_deletions(
                args.repo_root, config, args.check_pr_deletions, args.head_ref
            )
        )
    if args.disable:
        repo = args.repo or os.environ.get("GITHUB_REPOSITORY", "")
        if not repo:
            print(
                "GITHUB_REPOSITORY or --repo is required for --disable", file=sys.stderr
            )
            return 2
        for line in disable_retired_workflows(repo, config, dry_run=args.dry_run):
            print(line)
    if findings:
        for item in findings:
            print(item, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
