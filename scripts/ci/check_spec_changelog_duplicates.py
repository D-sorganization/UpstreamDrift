#!/usr/bin/env python3
"""Fail when SPEC.md's changelog gains a *new* duplicate version number.

Usage::

    python3 scripts/ci/check_spec_changelog_duplicates.py

``SPEC.md``'s Section 12 changelog is hot-prepended by every pull request, so
it is a serialization point that every concurrent branch collides on.  The
conventional resolution -- keep both sides' rows -- is right for the *prose*
but silently wrong for the *version number*, because two branches routinely
pick the same next-free value before either has merged.  Nothing detected
that, so the duplicates accumulated: at the time this guard was written 507
rows carried only 433 distinct versions, with 54 numbers used more than once.

A duplicated version makes the changelog ambiguous as a record -- "what
changed in 1.0.629" stops having one answer -- and it defeats the ordering
the table relies on.

This is a **ratchet**, not a cleanup.  The 54 pre-existing duplicates are
recorded in ``scripts/config/spec_changelog_duplicate_baseline.json`` and
tolerated at exactly their recorded multiplicity; anything beyond that fails.
Historical rows are debt to reconcile deliberately, but a new collision is a
merge accident and should be caught while the author still has the context to
fix it.

Pure stdlib so it can run before any project dependencies are installed.
"""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "SPEC.md"
BASELINE_PATH = (
    REPO_ROOT / "scripts" / "config" / "spec_changelog_duplicate_baseline.json"
)

#: A changelog row: ``| YYYY-MM-DD | 1.0.N | text |``.
_ROW = re.compile(r"^\| \d{4}-\d{2}-\d{2} \| (\d+\.\d+\.\d+) \|", re.MULTILINE)


def _version_counts(spec_text: str) -> collections.Counter[str]:
    """Count how many changelog rows carry each version."""
    return collections.Counter(_ROW.findall(spec_text))


def _load_baseline(path: Path) -> dict[str, int]:
    """Read the tolerated duplicate multiplicities, or an empty allowance."""
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): int(v) for k, v in payload.get("duplicates", {}).items()}


def _sort_key(version: str) -> tuple[int, ...]:
    """Order versions numerically rather than lexically."""
    return tuple(int(part) for part in version.split("."))


def main() -> int:
    """Return 0 when no version exceeds its tolerated row count."""
    if not SPEC_PATH.is_file():
        print(f"SPEC.md not found at {SPEC_PATH}", file=sys.stderr)
        return 1

    counts = _version_counts(SPEC_PATH.read_text(encoding="utf-8"))
    if not counts:
        print(
            "No changelog rows matched; the table format may have changed.",
            file=sys.stderr,
        )
        return 1

    baseline = _load_baseline(BASELINE_PATH)
    offenders = {
        version: (count, baseline.get(version, 1))
        for version, count in counts.items()
        if count > baseline.get(version, 1)
    }
    if not offenders:
        return 0

    print(
        "SPEC.md changelog: a version number is used by more rows than allowed.",
        file=sys.stderr,
    )
    for version in sorted(offenders, key=_sort_key):
        found, allowed = offenders[version]
        print(f"  {version}: {found} rows, allowed {allowed}", file=sys.stderr)
    print(
        "\nTwo branches almost certainly picked the same next-free version before\n"
        "either merged. Renumber this branch's row to the next value nobody has\n"
        "used, keeping every row's prose. Do not add the collision to\n"
        f"{BASELINE_PATH.relative_to(REPO_ROOT).as_posix()} -- that file records historical\n"
        "debt and must only ever shrink.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
