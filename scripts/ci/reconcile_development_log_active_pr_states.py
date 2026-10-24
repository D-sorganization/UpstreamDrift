#!/usr/bin/env python3
"""Reconcile active development-log rows against GitHub pull request truth.

Postcondition: every active entry whose PR field names only merged pull requests
is transitioned to ``shipped``; entries naming only closed (unmerged) pull
requests are marked ``abandoned``. Entries with open pull requests or unknown
numbers are left unchanged.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import NamedTuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
REPO_DEFAULT = "D-sorganization/UpstreamDrift"
PR_REF = re.compile(r"#(\d+)")
STATE_LINE = re.compile(r"^(-\s+\*\*State:\*\*\s*)(.+?)(\s*)$")
SHIPPED_LINE = re.compile(r"^-\s+\*\*Shipped:\*\*")
NEXT_STEP_LINE = re.compile(r"^-\s+\*\*Next step:\*\*")


class PrRecord(NamedTuple):
    state: str
    merged_at: str | None


def _repo_root(value: Path) -> Path:
    root = value.resolve()
    if not root.is_dir():
        raise ValueError(f"repo root does not exist: {root}")
    return root


def _load_devlog(root: Path) -> Path:
    from shared_scripts.development_log_schema import resolve_canonical_devlog_path

    path = resolve_canonical_devlog_path(root)
    if not path.is_file():
        raise FileNotFoundError(f"development log not found: {path}")
    return path


def _pr_numbers(text: str) -> list[int]:
    return [int(match) for match in PR_REF.findall(text)]


def fetch_pr_record(repo: str, number: int) -> PrRecord | None:
    """Return pull request metadata, or ``None`` when the number does not resolve."""
    proc = subprocess.run(
        [
            "gh",
            "pr",
            "view",
            str(number),
            "--repo",
            repo,
            "--json",
            "state,mergedAt",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    payload = json.loads(proc.stdout)
    state = str(payload.get("state", "")).upper()
    merged_at = payload.get("mergedAt")
    return PrRecord(state=state, merged_at=merged_at or None)


def classify_entry(prs: list[int], records: dict[int, PrRecord | None]) -> str:
    """Return ``merged``, ``closed``, ``open``, ``unknown``, or ``no_pr``.

    Issue numbers sometimes appear in the PR field; unresolvable ``gh pr view``
    results are ignored so merged delivery PRs can still ship the entry.
    """
    if not prs:
        return "no_pr"
    states: list[str] = []
    for number in prs:
        record = records.get(number)
        if record is None:
            continue
        states.append(record.state)
    if not states:
        return "unknown"
    if any(state == "OPEN" for state in states):
        return "open"
    if all(state == "MERGED" for state in states):
        return "merged"
    if all(state == "CLOSED" for state in states):
        return "closed"
    if any(state == "MERGED" for state in states):
        return "merged"
    return "unknown"


def _merged_ship_date(prs: list[int], records: dict[int, PrRecord | None]) -> str:
    dates: list[str] = []
    for number in prs:
        record = records[number]
        if record and record.merged_at:
            dates.append(record.merged_at[:10])
    return max(dates) if dates else date.today().isoformat()


def reconcile_active_pr_states(
    content: str,
    *,
    records: dict[int, PrRecord | None],
    today: date | None = None,
) -> tuple[str, dict[str, int]]:
    """Return updated log text and action counts keyed by decision."""
    today = today or date.today()
    lines = content.splitlines(keepends=True)
    output: list[str] = []
    counts = {"merged": 0, "closed": 0, "skipped": 0}
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.startswith("### DL-"):
            output.append(line)
            index += 1
            continue
        block_start = index
        index += 1
        while (
            index < len(lines)
            and not lines[index].startswith("### DL-")
            and not (
                lines[index].startswith("## ") and not lines[index].startswith("### ")
            )
        ):
            index += 1
        block = lines[block_start:index]
        state_field = next(
            (raw for raw in block if raw.lstrip().startswith("- **State:**")),
            "",
        )
        state_match = STATE_LINE.match(state_field.rstrip("\r\n"))
        current_state = (
            state_match.group(2).strip("`").split()[0].lower() if state_match else ""
        )
        from shared_scripts.development_log_schema import ACTIVE_STATES

        if current_state not in ACTIVE_STATES:
            counts["skipped"] += 1
            output.extend(block)
            continue

        pr_field = next(
            (raw for raw in block if raw.lstrip().startswith("- **PR:**")),
            "",
        )
        prs = _pr_numbers(pr_field)
        decision = classify_entry(prs, records)
        if decision == "merged":
            counts["merged"] += 1
            block = _apply_terminal_state(
                block,
                new_state="shipped",
                shipped_on=_merged_ship_date(prs, records),
            )
        elif decision == "closed":
            counts["closed"] += 1
            block = _apply_terminal_state(
                block,
                new_state="abandoned",
                shipped_on=today.isoformat(),
            )
        else:
            counts["skipped"] += 1
        output.extend(block)
    return "".join(output), counts


def _apply_terminal_state(
    block: list[str], *, new_state: str, shipped_on: str
) -> list[str]:
    updated: list[str] = []
    state_done = False
    shipped_done = any(SHIPPED_LINE.match(line.rstrip("\r\n")) for line in block)
    for line in block:
        stripped = line.rstrip("\r\n")
        if NEXT_STEP_LINE.match(stripped):
            continue
        state_match = STATE_LINE.match(stripped)
        if state_match and not state_done:
            updated.append(f"{state_match.group(1)}{new_state}{state_match.group(3)}\n")
            state_done = True
            if new_state == "shipped" and not shipped_done:
                updated.append(f"- **Shipped:** {shipped_on}\n")
                shipped_done = True
            continue
        updated.append(line)
    if state_done and new_state == "shipped" and not shipped_done:
        updated.append(f"- **Shipped:** {shipped_on}\n")
    return updated


def collect_pr_numbers(content: str) -> list[int]:
    from shared_scripts.development_log_schema import parse_entries

    numbers: set[int] = set()
    for entry in parse_entries(content):
        if not entry.is_active:
            continue
        numbers.update(_pr_numbers(entry.fields.get("PR", "")))
    return sorted(numbers)


def load_pr_records(repo: str, numbers: list[int]) -> dict[int, PrRecord | None]:
    return {number: fetch_pr_record(repo, number) for number in numbers}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--github-repo", default=REPO_DEFAULT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts only; do not write the development log.",
    )
    args = parser.parse_args(argv)

    root = _repo_root(args.repo_root)
    log_path = _load_devlog(root)
    original = log_path.read_text(encoding="utf-8")
    pr_numbers = collect_pr_numbers(original)
    if not pr_numbers:
        print("No active PR references to reconcile.")
        return 0

    records = load_pr_records(args.github_repo, pr_numbers)
    updated, counts = reconcile_active_pr_states(original, records=records)
    print(json.dumps({"pr_refs": len(pr_numbers), "actions": counts}, indent=2))
    if args.dry_run or updated == original:
        return 0

    log_path.write_text(updated, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
