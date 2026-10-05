#!/usr/bin/env python3
"""Enqueue green, armed PRs that the merge queue never picked up (#2018).

When auto-merge is armed while a PR's required checks are still running,
GitHub sometimes never adds the PR to the merge queue once they pass. The PR
then sits green, ``mergeable_state == clean`` and armed until someone re-arms
it. This scans one repository and enqueues each such PR through
``automerge_guard.enqueue_stalled_pr``, so every reviewer hold still applies.

Network budget per run: one REST list of open PRs, then two REST reads and at
most one GraphQL ``enqueuePullRequest`` per stalled PR, capped by ``--limit``.
Vendor this file next to ``automerge_guard.py``; it imports it by path.
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

# Expose this directory only while importing the guard. A lasting sys.path
# entry would shadow any same-named top-level package for the rest of the
# process (UpstreamDrift's scripts/motion_capture hid src/motion_capture).
_HERE = str(Path(__file__).resolve().parent)
sys.path.insert(0, _HERE)
try:
    import automerge_guard  # noqa: E402
finally:
    sys.path.remove(_HERE)

logger = logging.getLogger("requeue_stalled_merges")

CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]
Enqueue = Callable[[str, int], tuple[bool, str]]

#: Upper bound on enqueues per run, so a bad read can never mass-enqueue.
DEFAULT_LIMIT = 10
_REPO_SLUG = re.compile(r"^[\w.-]+/[\w.-]+$")
_OPEN_ARMED_JQ = ".[] | select(.draft | not) | select(.auto_merge != null) | .number"
_QUEUE_EVENTS_JQ = (
    '.[] | select(.event == "added_to_merge_queue" '
    'or .event == "removed_from_merge_queue") | .event'
)


@dataclass(frozen=True)
class RequeueReport:
    """Outcome of one scan. ``stalled`` is every PR that met the conditions."""

    stalled: tuple[int, ...]
    enqueued: tuple[int, ...]
    failed: tuple[int, ...]
    skipped_over_limit: int = 0


def _run(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(cmd), capture_output=True, text=True, check=False)


def _gh_lines(run: CommandRunner, args: Sequence[str]) -> list[str]:
    proc = run(["gh", "api", *args])
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "gh failed")
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def open_armed_prs(repo: str, run: CommandRunner) -> list[int]:
    """Open, non-draft PRs with auto-merge armed, in one REST call.

    A PR already in the merge queue reads ``auto_merge: null``, so it never
    appears here; :func:`is_stalled` checks the timeline as well regardless.
    """
    lines = _gh_lines(
        run, [f"repos/{repo}/pulls?state=open&per_page=100", "--jq", _OPEN_ARMED_JQ]
    )
    return [int(line) for line in lines]


def is_stalled(repo: str, pr: int, run: CommandRunner) -> bool:
    """True when ``pr`` is clean and its latest merge-queue event is not an add."""
    state = _gh_lines(run, [f"repos/{repo}/pulls/{pr}", "--jq", ".mergeable_state"])
    if state != ["clean"]:
        return False
    events = _gh_lines(
        run,
        [
            f"repos/{repo}/issues/{pr}/timeline?per_page=100",
            "--paginate",
            "--jq",
            _QUEUE_EVENTS_JQ,
        ],
    )
    return not events or events[-1] != "added_to_merge_queue"


#: GitHub's reply when the PR was queued between our scan and our call.
_ALREADY_QUEUED = re.compile(r"already (?:queued|in the merge queue)", re.IGNORECASE)


def _guard_enqueue(repo: str, pr: int) -> tuple[bool, str]:
    """Enqueue through the guard; losing the race to GitHub's own enqueue is ok."""
    result = automerge_guard.enqueue_stalled_pr(repo, pr)
    if not result.armed and _ALREADY_QUEUED.search(result.detail):
        return True, f"already queued: {result.detail}"
    return result.armed, result.detail


def requeue_stalled(
    repo: str,
    *,
    run: CommandRunner = _run,
    enqueue: Enqueue = _guard_enqueue,
    dry_run: bool = False,
    limit: int = DEFAULT_LIMIT,
) -> RequeueReport:
    """Scan ``repo`` and enqueue up to ``limit`` stalled PRs.

    Preconditions: ``repo`` is ``owner/name``; ``limit`` >= 1.
    Postconditions: ``enqueue`` is never called in ``dry_run``, never more than
    ``limit`` times, and only for PRs :func:`is_stalled` accepted. A failure
    listing open PRs raises; a failure on one PR is reported and the scan goes on.
    """
    if not _REPO_SLUG.match(repo):
        raise ValueError(f"repo must be owner/name, got {repo!r}")
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")
    stalled: list[int] = []
    for pr in open_armed_prs(repo, run):
        try:
            if is_stalled(repo, pr, run):
                stalled.append(pr)
        except RuntimeError as exc:
            logger.warning("Could not read %s#%s: %s", repo, pr, exc)
    if dry_run:
        for pr in stalled:
            logger.info("[dry-run] would enqueue %s#%s", repo, pr)
        return RequeueReport(stalled=tuple(stalled), enqueued=(), failed=())
    enqueued: list[int] = []
    failed: list[int] = []
    for pr in stalled[:limit]:
        ok, detail = enqueue(repo, pr)
        (enqueued if ok else failed).append(pr)
        logger.info("%s#%s: %s", repo, pr, detail)
    report = RequeueReport(
        stalled=tuple(stalled),
        enqueued=tuple(enqueued),
        failed=tuple(failed),
        skipped_over_limit=max(0, len(stalled) - limit),
    )
    assert len(report.enqueued) + len(report.failed) <= limit
    return report


def exit_code(report: RequeueReport) -> int:
    """0 when every attempted enqueue succeeded, 1 when any failed."""
    return 1 if report.failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: ``requeue_stalled_merges.py <owner/repo> [--dry-run] [--limit N]``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("repo", help="owner/name")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        report = requeue_stalled(args.repo, dry_run=args.dry_run, limit=args.limit)
    except (RuntimeError, ValueError) as exc:
        logger.error("requeue scan failed: %s", exc)
        return 1
    logger.info(
        "stalled=%s enqueued=%s failed=%s skipped_over_limit=%d",
        list(report.stalled),
        list(report.enqueued),
        list(report.failed),
        report.skipped_over_limit,
    )
    return exit_code(report)


if __name__ == "__main__":
    sys.exit(main())
