#!/usr/bin/env python3
"""Validate UpstreamDrift merge-queue ruleset settings against a committed snapshot.

Postcondition: ``check_against_live`` returns ``(True, message)`` only when the
live GitHub ruleset's ``merge_queue`` parameters match the snapshot exactly for
every key present in the snapshot.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]

DEFAULT_REPO = "D-sorganization/UpstreamDrift"
DEFAULT_SNAPSHOT = (
    Path(__file__).resolve().parents[1]
    / "docs/operations/settings-snapshots/UpstreamDrift-merge-queue-target.json"
)

_MERGE_QUEUE_KEYS = (
    "merge_method",
    "max_entries_to_build",
    "min_entries_to_merge",
    "max_entries_to_merge",
    "min_entries_to_merge_wait_minutes",
    "grouping_strategy",
    "check_response_timeout_minutes",
)


def load_snapshot(path: Path) -> dict[str, Any]:
    """Precondition: ``path`` exists and contains JSON with ``merge_queue``."""
    if not path.is_file():
        raise FileNotFoundError(f"snapshot not found: {path}")
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"snapshot root must be an object: {path}")
    if "merge_queue" not in data:
        raise ValueError(f"snapshot missing merge_queue block: {path}")
    if "ruleset_id" not in data:
        raise ValueError(f"snapshot missing ruleset_id: {path}")
    return data


def merge_queue_parameters(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Return the merge-queue parameter block from a snapshot."""
    block = snapshot["merge_queue"]
    if not isinstance(block, dict):
        raise TypeError("merge_queue must be an object")
    return dict(block)


def _live_merge_queue_parameters(ruleset: Mapping[str, Any]) -> dict[str, Any]:
    rules = ruleset.get("rules")
    if not isinstance(rules, list):
        raise ValueError("ruleset rules must be a list")
    for rule in rules:
        if isinstance(rule, dict) and rule.get("type") == "merge_queue":
            params = rule.get("parameters")
            if isinstance(params, dict):
                return dict(params)
    raise ValueError("ruleset has no merge_queue rule")


def diff_merge_queue(
    snapshot: Mapping[str, Any], live_ruleset: Mapping[str, Any]
) -> list[str]:
    """Return human-readable diffs; empty list means a match on tracked keys."""
    expected = merge_queue_parameters(snapshot)
    actual = _live_merge_queue_parameters(live_ruleset)
    snapshot_id = snapshot.get("ruleset_id")
    live_id = live_ruleset.get("id")
    if snapshot_id is not None and live_id is not None and snapshot_id != live_id:
        return [f"ruleset_id: snapshot={snapshot_id!r} live={live_id!r}"]
    diffs: list[str] = []
    for key in _MERGE_QUEUE_KEYS:
        if key not in expected:
            continue
        if actual.get(key) != expected[key]:
            diffs.append(f"{key}: snapshot={expected[key]!r} live={actual.get(key)!r}")
    return diffs


def _default_runner(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def fetch_live_ruleset(
    repo: str, ruleset_id: int, run: CommandRunner = _default_runner
) -> dict[str, Any]:
    """Fetch one repository ruleset via ``gh api``."""
    result = run(["gh", "api", f"repos/{repo}/rulesets/{ruleset_id}"])
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(f"gh api rulesets/{ruleset_id} failed: {stderr}")
    payload = json.loads(result.stdout or "{}")
    if not isinstance(payload, dict):
        raise ValueError("ruleset response must be a JSON object")
    return payload


def check_against_live(
    snapshot_path: Path,
    *,
    repo: str = DEFAULT_REPO,
    run: CommandRunner = _default_runner,
) -> tuple[bool, str]:
    """Compare ``snapshot_path`` to the live GitHub ruleset (fail closed on errors)."""
    snapshot = load_snapshot(snapshot_path)
    ruleset_id = int(snapshot["ruleset_id"])
    live = fetch_live_ruleset(repo, ruleset_id, run=run)
    diffs = diff_merge_queue(snapshot, live)
    if diffs:
        return False, "live settings differ from snapshot: " + "; ".join(diffs)
    return True, "live settings match snapshot"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=DEFAULT_SNAPSHOT,
        help="Path to the committed merge-queue settings snapshot JSON",
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help="owner/name repository slug for live checks",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Compare the snapshot to the live GitHub ruleset (requires gh + token)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.live:
        load_snapshot(args.snapshot)
        print(f"Snapshot OK: {args.snapshot}")
        return 0
    ok, detail = check_against_live(args.snapshot, repo=args.repo)
    print(detail)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
