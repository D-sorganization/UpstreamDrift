#!/usr/bin/env python3
"""Fail when SPEC.md's changelog gains a new duplicate version *or* entry.

Usage::

    python3 scripts/ci/check_spec_changelog_duplicates.py

``SPEC.md``'s Section 12 changelog is hot-prepended by every pull request, so
it is a serialization point that every concurrent branch collides on. The
conventional resolution -- keep both sides' rows -- is right for the prose but
wrong in two distinct ways that nothing detected:

1. **Two branches pick the same next-free version** before either merges, so
   one number ends up on two different entries and "what changed in 1.0.629"
   stops having one answer.
2. **A branch renumbers to dodge a collision and then merges the branch it was
   dodging**, so the *same* entry is logged twice under two numbers. The
   version check cannot see this, because the numbers genuinely differ.

The second was found only when a reviewer noticed two 4437-character rows that
were byte-identical apart from their version. Both checks are therefore
**ratchets**: what already exists is recorded in
``scripts/config/spec_changelog_duplicate_baseline.json`` and tolerated at
exactly its present multiplicity, while anything new fails with the offender
named. Historical rows are debt to reconcile deliberately; a new collision is
a merge accident and should be caught while the author still has the context.

Pure stdlib so it can run before any project dependencies are installed.
"""

from __future__ import annotations

import collections
import hashlib
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "SPEC.md"
BASELINE_PATH = (
    REPO_ROOT / "scripts" / "config" / "spec_changelog_duplicate_baseline.json"
)

#: A changelog row's version: ``| YYYY-MM-DD | 1.0.N | ...``.
_ROW = re.compile(r"^\| \d{4}-\d{2}-\d{2} \| (\d+\.\d+\.\d+) \|", re.MULTILINE)

#: A full row, so the body can be fingerprinted as well as the number.
_FULL_ROW = re.compile(
    r"^\| \d{4}-\d{2}-\d{2} \| (\d+\.\d+\.\d+) \| (.+?) \|\s*$", re.MULTILINE
)

#: Bodies shorter than this repeat innocently ("Version bump."); only
#: substantial prose is treated as a fingerprint.
_MIN_BODY_CHARS = 80


def _load_document(path: Path) -> dict[str, object]:
    """Read the baseline document, or an empty allowance when it is absent."""
    if not path.is_file():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _version_counts(spec_text: str) -> collections.Counter[str]:
    """Count how many changelog rows carry each version."""
    return collections.Counter(_ROW.findall(spec_text))


def _duplicate_bodies(spec_text: str) -> dict[str, list[str]]:
    """Group versions whose changelog prose is byte-identical."""
    groups: dict[str, list[str]] = {}
    for version, body in _FULL_ROW.findall(spec_text):
        text = body.strip()
        if len(text) < _MIN_BODY_CHARS:
            continue
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        groups.setdefault(digest, []).append(version)
    return {digest: sorted(v) for digest, v in groups.items() if len(v) > 1}


def _sort_key(version: str) -> tuple[int, ...]:
    """Order versions numerically rather than lexically."""
    return tuple(int(part) for part in version.split("."))


def _advice(what: str) -> str:
    """Closing guidance shared by both failure paths."""
    relative = BASELINE_PATH.relative_to(REPO_ROOT).as_posix()
    return (
        f"\n{what}\nDo not add it to {relative} -- that file records historical\n"
        "debt and must only ever shrink."
    )


def _report_versions(offenders: dict[str, tuple[int, int]]) -> None:
    """Explain a version used by more rows than the baseline allows."""
    print(
        "SPEC.md changelog: a version number is used by more rows than allowed.",
        file=sys.stderr,
    )
    for version in sorted(offenders, key=_sort_key):
        found, allowed = offenders[version]
        print(f"  {version}: {found} rows, allowed {allowed}", file=sys.stderr)
    print(
        _advice(
            "Two branches almost certainly picked the same next-free version\n"
            "before either merged. Renumber this branch's row to the next value\n"
            "nobody has used, keeping every row's prose."
        ),
        file=sys.stderr,
    )


def _report_bodies(repeated: dict[str, list[str]]) -> None:
    """Explain one entry logged under two version numbers."""
    print(
        "SPEC.md changelog: one entry is logged twice under different versions.",
        file=sys.stderr,
    )
    for digest in sorted(repeated):
        print(f"  {digest}: {', '.join(repeated[digest])}", file=sys.stderr)
    print(
        _advice(
            "A hot-prepend conflict was resolved by keeping BOTH rows where one\n"
            "side was already a renumbered copy of the other, so one change is\n"
            "now logged twice. Keep one row and drop the duplicate."
        ),
        file=sys.stderr,
    )


def main() -> int:
    """Return 0 when no version and no entry body is duplicated beyond baseline."""
    if not SPEC_PATH.is_file():
        print(f"SPEC.md not found at {SPEC_PATH}", file=sys.stderr)
        return 1

    spec_text = SPEC_PATH.read_text(encoding="utf-8")
    counts = _version_counts(spec_text)
    if not counts:
        print(
            "No changelog rows matched; the table format may have changed.",
            file=sys.stderr,
        )
        return 1

    document = _load_document(BASELINE_PATH)
    allowed_versions = {
        str(key): int(value)
        for key, value in (document.get("duplicates") or {}).items()
    }
    offenders = {
        version: (count, allowed_versions.get(version, 1))
        for version, count in counts.items()
        if count > allowed_versions.get(version, 1)
    }
    if offenders:
        _report_versions(offenders)
        return 1

    allowed_bodies = {
        str(key): [str(item) for item in value]
        for key, value in (document.get("duplicate_text") or {}).items()
    }
    repeated = {
        digest: versions
        for digest, versions in _duplicate_bodies(spec_text).items()
        if versions != allowed_bodies.get(digest)
    }
    if repeated:
        _report_bodies(repeated)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
