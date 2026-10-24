#!/usr/bin/env python3
"""Apply merge-queue parameters from a committed settings snapshot via ``gh api``.

Precondition: the caller has permission to update repository rulesets. This script
never runs in CI; operators run it once after the snapshot PR merges.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECK_MODULE = REPO_ROOT / "scripts" / "check_fleet_merge_settings.py"


def _load_check_module() -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "check_fleet_merge_settings", CHECK_MODULE
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {CHECK_MODULE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fetch_ruleset(repo: str, ruleset_id: int) -> dict[str, Any]:
    result = subprocess.run(
        ["gh", "api", f"repos/{repo}/rulesets/{ruleset_id}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or "").strip() or "gh api failed")
    payload = json.loads(result.stdout or "{}")
    if not isinstance(payload, dict):
        raise ValueError("ruleset response must be an object")
    return payload


def _apply_merge_queue(
    ruleset: dict[str, Any], parameters: dict[str, Any]
) -> dict[str, Any]:
    rules = ruleset.get("rules")
    if not isinstance(rules, list):
        raise ValueError("ruleset rules must be a list")
    updated: list[Any] = []
    found = False
    for rule in rules:
        if isinstance(rule, dict) and rule.get("type") == "merge_queue":
            found = True
            merged = dict(rule)
            merged["parameters"] = {**dict(rule.get("parameters") or {}), **parameters}
            updated.append(merged)
        else:
            updated.append(rule)
    if not found:
        raise ValueError("ruleset has no merge_queue rule to update")
    body = {key: ruleset[key] for key in ruleset if key != "id"}
    body["rules"] = updated
    return body


def apply_snapshot(snapshot_path: Path, *, repo: str, dry_run: bool) -> None:
    check = _load_check_module()
    snapshot = check.load_snapshot(snapshot_path)
    ruleset_id = int(snapshot["ruleset_id"])
    parameters = check.merge_queue_parameters(snapshot)
    live = _fetch_ruleset(repo, ruleset_id)
    body = _apply_merge_queue(live, parameters)
    if dry_run:
        print(json.dumps(body, indent=2))
        return
    payload = json.dumps(body)
    result = subprocess.run(
        [
            "gh",
            "api",
            "--method",
            "PUT",
            f"repos/{repo}/rulesets/{ruleset_id}",
            "--input",
            "-",
        ],
        input=payload,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or "").strip() or "ruleset update failed")
    print(f"Updated ruleset {ruleset_id} for {repo}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--repo", default="D-sorganization/UpstreamDrift")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the PUT body without calling the GitHub API",
    )
    args = parser.parse_args(argv)
    apply_snapshot(args.snapshot, repo=args.repo, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
