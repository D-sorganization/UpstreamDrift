#!/usr/bin/env python3
"""Generate NSX-00..75 salvage inventory markdown (stdout)."""

from __future__ import annotations

import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

REPO = "D-sorganization/UpstreamDrift"
MAIN = "origin/main"


def main() -> None:
    nsx_issue = _find_issue_map()
    nsx_merge = _find_merged_map()
    pr_by_issue = _find_open_prs_by_issue()

    deferred = set(range(70, 76))
    absorbed = {
        n
        for n in range(76)
        if n not in nsx_issue and n not in nsx_merge and n not in deferred
    }

    print("# NSX program salvage tracker (retire PR #12634)\n")
    print(
        "Archive branch: `feat/nsx-consolidated-wave15-19`. "
        "Parent science epic: #12593.\n"
    )
    _print_inventory(nsx_issue, nsx_merge, pr_by_issue, deferred, absorbed)


def _find_issue_map() -> dict[int, int]:
    nsx_issue: dict[int, int] = {}
    root = Path("docs/research/loop_nullspace")
    if root.is_dir():
        for path in root.rglob("*.md"):
            text = path.read_text(encoding="utf-8")
            for m in re.finditer(r"NSX-(\d+)[^\n#]*\(#(\d+)", text):
                nsx_issue.setdefault(int(m.group(1)), int(m.group(2)))
    for num in range(12549, 12594):
        proc = subprocess.run(
            [
                "gh",
                "issue",
                "view",
                str(num),
                "--repo",
                REPO,
                "--json",
                "title,state",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            continue
        data = json.loads(proc.stdout)
        match = re.search(r"\[NSX-(\d+)\]", data["title"])
        if match:
            nsx_issue[int(match.group(1))] = num
    return nsx_issue


def _find_merged_map() -> dict[int, str]:
    log = subprocess.run(
        ["git", "log", MAIN, "--oneline", "-10000"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    nsx_merge: dict[int, str] = {}
    for line in log.splitlines():
        if "#12634" in line:
            continue
        for m in re.finditer(r"NSX-(\d+)", line, re.IGNORECASE):
            nsx_merge.setdefault(int(m.group(1)), line)
    return nsx_merge


def _find_open_prs_by_issue() -> dict[int, list[dict[str, object]]]:
    open_prs = json.loads(
        subprocess.run(
            [
                "gh",
                "pr",
                "list",
                "--repo",
                REPO,
                "--state",
                "open",
                "--limit",
                "150",
                "--json",
                "number,title,mergeable,headRefName",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    pr_by_issue: dict[int, list[dict[str, object]]] = defaultdict(list)
    for pr in open_prs:
        if pr["number"] == 12634:
            continue
        for m in re.finditer(r"#(\d+)", str(pr["title"])):
            pr_by_issue[int(m.group(1))].append(pr)
    return pr_by_issue


def _print_inventory(
    nsx_issue: dict[int, int],
    nsx_merge: dict[int, str],
    pr_by_issue: dict[int, list[dict[str, object]]],
    deferred: set[int],
    absorbed: set[int],
) -> None:
    for nsx in range(76):
        issue = nsx_issue.get(nsx)
        issue_ref = f"#{issue}" if issue else "—"
        if nsx in nsx_merge:
            line = nsx_merge[nsx]
            pr_m = re.search(r"#(\d+)", line)
            pr_ref = f"#{pr_m.group(1)}" if pr_m else "?"
            state = f"merged on main ({pr_ref})"
            checked = "x"
        elif issue:
            iv = subprocess.run(
                [
                    "gh",
                    "issue",
                    "view",
                    str(issue),
                    "--repo",
                    REPO,
                    "--json",
                    "state",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            istate = json.loads(iv.stdout)["state"] if iv.returncode == 0 else "?"
            oprs = pr_by_issue.get(issue, [])
            if oprs:
                p = oprs[0]
                state = (
                    f"open slice PR #{p['number']} "
                    f"({p['mergeable']}, `{p['headRefName']}`)"
                )
            elif istate == "OPEN":
                state = f"open issue #{issue}, no green slice PR"
            elif istate == "CLOSED":
                state = "issue closed (likely via consolidate #13033)"
                checked = "x"
            else:
                state = f"issue {istate}"
            checked = "x" if nsx in nsx_merge else " "
        elif nsx in deferred:
            state = "research spike — deferred"
            checked = " "
        elif nsx in absorbed:
            state = "absorbed in NSX-00 / no separate child issue"
            checked = "x"
        else:
            state = "only on stale #12634 branch — cherry-pick if still needed"
            checked = " "

        rank = _rank(nsx, state)
        print(f"- [{checked}] **NSX-{nsx:02d}** ({issue_ref}) — rank {rank} — {state}")


def _rank(nsx: int, state: str) -> str:
    if "deferred" in state or nsx >= 80:
        return "D"
    if nsx in (4, 22, 24, 45, 55, 60):
        return "A"
    if "merged" in state or "closed" in state or "absorbed" in state:
        return "done"
    if nsx in (25, 26, 31, 40, 43, 44, 61, 62):
        return "B"
    return "C"


if __name__ == "__main__":
    main()
