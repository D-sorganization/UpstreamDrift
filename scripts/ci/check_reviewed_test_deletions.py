#!/usr/bin/env python3
"""Gate deleted Python test files on a recorded, reviewed waiver.

CI refuses to run a pull request that deletes a file under ``tests/`` unless
that deletion is recorded in ``scripts/config/reviewed_test_deletions.json``.
The waiver lands in the same diff as the deletion, so the reviewer who
approves the deletion also approves the record of why coverage did not
silently vanish -- which is the whole point of the guard in
``.github/workflows/ci-standard.yml`` (#7368).

A waiver names a ``replacement`` test that must exist in the working tree, so
"this test was superseded" is a checked claim rather than an assertion. A test
that is genuinely retired rather than replaced sets ``replacement`` to ``null``
and says why in ``reason``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

SCHEMA_VERSION = 1
REQUIRED_FIELDS = ("path", "replacement", "reason", "tracked_in", "expires_on")
DEFAULT_CONFIG = Path("scripts/config/reviewed_test_deletions.json")


@dataclass(frozen=True)
class ReviewedDeletion:
    """One reviewed deletion of a Python test file."""

    path: str
    replacement: str | None
    reason: str
    tracked_in: str
    expires_on: date


def _load_manifest(path: Path) -> dict[str, object]:
    """Read and validate the top-level waiver manifest object."""
    if not path.exists():
        raise FileNotFoundError(f"reviewed-test-deletion file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8") or "{}")
    if not isinstance(raw, dict):
        raise ValueError("reviewed-test-deletion file must contain a JSON object")
    schema_version = raw.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {schema_version}")
    return raw


def _parse_entry(item: object) -> ReviewedDeletion:
    """Parse and validate one reviewed-deletion entry."""
    if not isinstance(item, dict):
        raise ValueError("each reviewed-deletion entry must be a mapping")
    missing_fields = [field for field in REQUIRED_FIELDS if field not in item]
    if missing_fields:
        raise ValueError(f"missing reviewed-deletion field: {missing_fields[0]}")
    replacement_raw = item["replacement"]
    replacement = None if replacement_raw is None else str(replacement_raw).strip()
    try:
        entry = ReviewedDeletion(
            path=str(item["path"]).strip(),
            replacement=replacement or None,
            reason=str(item["reason"]).strip(),
            tracked_in=str(item["tracked_in"]).strip(),
            expires_on=date.fromisoformat(str(item["expires_on"]).strip()),
        )
    except ValueError as exc:
        raise ValueError("expires_on must be an ISO date") from exc

    _validate_entry_metadata(entry)
    return entry


def _validate_entry_metadata(entry: ReviewedDeletion) -> None:
    """Reject entries whose metadata is empty or obviously unreviewable."""
    if not entry.path:
        raise ValueError("path must not be empty")
    if not entry.reason:
        raise ValueError(f"reason must not be empty for {entry.path}")
    if not entry.tracked_in:
        raise ValueError(f"tracked_in must not be empty for {entry.path}")


def load_entries(config: Path) -> dict[str, ReviewedDeletion]:
    """Return the reviewed deletions keyed by deleted path."""
    manifest = _load_manifest(config)
    raw_entries = manifest.get("reviewed_deletions", [])
    if not isinstance(raw_entries, list):
        raise ValueError("reviewed_deletions must be a list")
    entries = [_parse_entry(item) for item in raw_entries]
    return {entry.path: entry for entry in entries}


def _read_deleted_paths(list_file: Path | None, extra: list[str]) -> list[str]:
    """Collect deleted test paths from a diff listing file and/or arguments."""
    paths = [candidate.strip() for candidate in extra if candidate.strip()]
    if list_file is not None:
        text = list_file.read_text(encoding="utf-8") if list_file.exists() else ""
        paths.extend(line.strip() for line in text.splitlines() if line.strip())
    seen: dict[str, None] = {}
    for candidate in paths:
        seen.setdefault(candidate.replace("\\", "/"), None)
    return list(seen)


def check_deletions(
    deleted: list[str],
    entries: dict[str, ReviewedDeletion],
    repo_root: Path,
    today: date,
) -> list[str]:
    """Return one failure message per deletion that is not properly reviewed."""
    failures: list[str] = []
    for path in deleted:
        entry = entries.get(path)
        if entry is None:
            failures.append(
                f"{path}: deleted with no reviewed-deletion record. Add an entry to "
                f"{DEFAULT_CONFIG.as_posix()} naming the replacement test and the "
                "tracking issue, or restore the file."
            )
            continue
        if entry.expires_on < today:
            failures.append(
                f"{path}: reviewed-deletion record expired on "
                f"{entry.expires_on.isoformat()}; refresh or remove it."
            )
            continue
        if entry.replacement is None:
            continue
        if not (repo_root / entry.replacement).exists():
            failures.append(
                f"{path}: declared replacement {entry.replacement} does not exist."
            )
    return failures


def _build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        help="Deleted test paths to validate (repository-relative).",
    )
    parser.add_argument(
        "--deleted-files",
        type=Path,
        default=None,
        help="File containing one deleted test path per line (git diff output).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"Waiver manifest (default: {DEFAULT_CONFIG.as_posix()}).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root used to resolve replacement paths.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate every deleted test path against the reviewed-deletion manifest."""
    args = _build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    config = args.config if args.config is not None else repo_root / DEFAULT_CONFIG

    deleted = _read_deleted_paths(args.deleted_files, list(args.paths))
    if not deleted:
        return 0

    try:
        entries = load_entries(config)
    except (OSError, ValueError) as exc:
        print(f"::error::Invalid reviewed-test-deletion manifest: {exc}")
        return 1

    failures = check_deletions(deleted, entries, repo_root, date.today())
    if failures:
        print(
            "::error::Deleted Python test files require review before CI can proceed."
        )
        for failure in failures:
            print(f"  {failure}")
        return 1

    print("Reviewed test deletions accepted:")
    for path in deleted:
        entry = entries[path]
        replacement = entry.replacement or "(retired, no replacement)"
        print(f"  {path} -> {replacement} ({entry.tracked_in})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
