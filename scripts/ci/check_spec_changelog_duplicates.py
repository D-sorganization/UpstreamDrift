#!/usr/bin/env python3
"""Validate SPEC.md's change log: PR-keyed rows, no duplicated entry.

Usage::

    python3 scripts/ci/check_spec_changelog_duplicates.py

History of this gate
--------------------
``SPEC.md``'s Section 12 change log used to carry a **serial spec version** in
every row, and every pull request had to claim the next free one. That made the
table a serialization point that every concurrent branch collided on, and it
produced two distinct defects this script was written to catch:

1. **Two branches pick the same next-free version** before either merges, so
   one number ends up on two entries and "what changed in 1.0.629" stops having
   one answer.
2. **A branch renumbers to dodge a collision and then merges the branch it was
   dodging**, so the *same* entry is logged twice under two numbers. The version
   check cannot see this, because the numbers genuinely differ.

Repository_Management#1520 removed the cause rather than continuing to police
the symptom: a row is now keyed by its **pull request**, which is unique by
construction, so defect 1 cannot happen — two pull requests can no longer pick
the same key. Defect 2 is *not* about keys at all; a copied row is still a
copied row, so the body-fingerprint ratchet below is kept exactly as it was.

What this enforces now
----------------------
* Every row parses as ``| YYYY-MM-DD | #<pr> | summary |`` (the shared
  ``spec_changelog`` contract, which also rejects a serial version sitting in
  the key column).
* No pull-request key is reused among rows dated on or after the migration
  cutover. Rows before it are exempt: several legitimately share one governing
  issue (one issue, several pull requests), so their recovered keys are not
  unique and never can be.
* No two rows share a byte-identical body beyond the recorded baseline.

The body check remains a **ratchet**: what already exists is recorded in
``scripts/config/spec_changelog_duplicate_baseline.json`` and tolerated at
exactly its present multiplicity, while anything new fails with the offender
named. Historical rows are debt to reconcile deliberately; a new duplicate is a
merge accident and should be caught while the author still has the context.

Pure stdlib so it can run before any project dependencies are installed.
"""

from __future__ import annotations

import collections
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "SPEC.md"
BASELINE_PATH = (
    REPO_ROOT / "scripts" / "config" / "spec_changelog_duplicate_baseline.json"
)

#: Bodies shorter than this repeat innocently ("Version bump."); only
#: substantial prose is treated as a fingerprint.
_MIN_BODY_CHARS = 80


def _load_spec_changelog() -> ModuleType:
    """Import the portable change-log contract by path.

    ``shared_scripts`` is not an importable package in every checkout, and this
    script deliberately runs before dependencies are installed.
    """
    module_path = REPO_ROOT / "shared_scripts" / "spec_changelog.py"
    spec = importlib.util.spec_from_file_location("ud_spec_changelog", module_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec: the module defines dataclasses, and dataclasses
    # resolves field types through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_document(path: Path) -> dict[str, Any]:
    """Read the baseline document, or an empty allowance when it is absent."""
    if not path.is_file():
        return {}
    loaded: Any = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _duplicate_bodies(rows: list[Any]) -> dict[str, list[str]]:
    """Group keys whose change-log prose is byte-identical."""
    groups: dict[str, list[str]] = collections.defaultdict(list)
    for row in rows:
        text = str(getattr(row, "summary", "")).strip()
        if len(text) < _MIN_BODY_CHARS:
            continue
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        groups[digest].append(str(getattr(row, "key", "")))
    return {digest: sorted(keys) for digest, keys in groups.items() if len(keys) > 1}


def _advice(what: str) -> str:
    """Closing guidance shared by the failure paths."""
    relative = BASELINE_PATH.relative_to(REPO_ROOT).as_posix()
    return (
        f"\n{what}\nDo not add it to {relative} -- that file records historical\n"
        "debt and must only ever shrink."
    )


def _report_contract(failures: list[str]) -> None:
    """Explain a row that does not satisfy the PR-keyed row contract."""
    print("SPEC.md changelog: row contract violated.", file=sys.stderr)
    for failure in failures:
        print(f"  {failure}", file=sys.stderr)
    print(
        "\nRows are keyed by pull request since Repository_Management#1520:\n"
        "  | YYYY-MM-DD | #<pr> | summary |\n"
        "Add exactly one row for your own pull request, do not put a serial\n"
        "spec version in the key column, and do not renumber anybody else's\n"
        "row. The Spec Version field is release-derived.",
        file=sys.stderr,
    )


def _report_bodies(repeated: dict[str, list[str]]) -> None:
    """Explain one entry logged twice."""
    print(
        "SPEC.md changelog: one entry is logged twice under different keys.",
        file=sys.stderr,
    )
    for digest in sorted(repeated):
        print(f"  {digest}: {', '.join(repeated[digest])}", file=sys.stderr)
    print(
        _advice(
            "A conflict was resolved by keeping BOTH rows where one side was\n"
            "already a copy of the other, so one change is now logged twice.\n"
            "Keep one row and drop the duplicate."
        ),
        file=sys.stderr,
    )


def main() -> int:
    """Return 0 when the change log satisfies the contract and the ratchet."""
    if not SPEC_PATH.is_file():
        print(f"SPEC.md not found at {SPEC_PATH}", file=sys.stderr)
        return 1

    changelog_module = _load_spec_changelog()
    spec_text = SPEC_PATH.read_text(encoding="utf-8")

    try:
        changelog = changelog_module.parse_changelog(spec_text)
    except changelog_module.SpecChangelogError as exc:
        print(f"SPEC.md changelog: {exc}", file=sys.stderr)
        return 1

    if not changelog.rows:
        print(
            "No changelog rows matched; the table format may have changed.",
            file=sys.stderr,
        )
        return 1

    failures = changelog_module.validate(changelog)
    if failures:
        _report_contract(failures)
        return 1

    document = _load_document(BASELINE_PATH)
    allowed_bodies = {
        str(key): [str(item) for item in value]
        for key, value in (document.get("duplicate_text") or {}).items()
    }
    repeated = {
        digest: keys
        for digest, keys in _duplicate_bodies(changelog.rows).items()
        if keys != allowed_bodies.get(digest)
    }
    if repeated:
        _report_bodies(repeated)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
