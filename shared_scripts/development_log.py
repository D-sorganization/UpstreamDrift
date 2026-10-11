#!/usr/bin/env python3
"""Canonical development-log schema validator and archiver.

A development log is an active-work dashboard, not a journal. This validator
enforces that format:

1. A required ``## Active`` section.
2. Entries keyed by ``DL-#<issue>`` (Repository_Management#1520) or ``DL-<serial>``
   (legacy).
3. A closed state vocabulary: proposed, in_progress, in_review, shipped, parked,
   abandoned.
4. Required fields: State, Owner, PR, Paths, Started, Last verified, Summary,
   plus Issue and Next step while the entry is active, Branch once in_progress /
   in_review, and Parked reason while parked.
5. No lingering angle-bracket placeholders.
6. A ceiling on active entries so work does not accumulate without shipping.
7. Shipped/abandoned entries can be archived to yearly files via ``--archive``.

See ``docs/templates/DEVELOPMENT_LOG.md`` for the canonical template.

Part of Repository_Management#1938: partitioned into ``development_log_schema.py``,
``development_log_validator.py``, and ``development_log.py``.
Portable: standard library only, copied fleet-wide next to ``handoff_validator.py``.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from types import ModuleType


def sibling(name: str) -> ModuleType:
    """Import a sibling fleet module by package, else by file path."""
    try:
        return importlib.import_module(f"shared_scripts.{name}")
    except ImportError:
        cached = sys.modules.get(f"_fleet_{name}")
        if cached is not None:
            return cached
        path = Path(__file__).with_name(f"{name}.py")
        spec = importlib.util.spec_from_file_location(f"_fleet_{name}", path)
        if spec is None or spec.loader is None:  # pragma: no cover - defensive
            raise
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module


_schema = sibling("development_log_schema")
_validator = sibling("development_log_validator")

# Re-export schema symbols
ACTIVE_ONLY_FIELDS = _schema.ACTIVE_ONLY_FIELDS
ACTIVE_STATES = _schema.ACTIVE_STATES
ARCHIVE_AFTER_DAYS = _schema.ARCHIVE_AFTER_DAYS
BASE_FIELDS = _schema.BASE_FIELDS
BRANCH_STATES = _schema.BRANCH_STATES
CANONICAL_RELATIVE_PATH = _schema.CANONICAL_RELATIVE_PATH
DevLogFinding = _schema.DevLogFinding
ENTRY_HEADING = _schema.ENTRY_HEADING
ENTRY_LOOKUP_HEADING = _schema.ENTRY_LOOKUP_HEADING
Entry = _schema.Entry
FIELD_LINE = _schema.FIELD_LINE
ISO_DATE = _schema.ISO_DATE
ISSUE_KEYED_ENTRY_ID = _schema.ISSUE_KEYED_ENTRY_ID
LEADING_DATE = _schema.LEADING_DATE
LEGACY_ENTRY_ID = _schema.LEGACY_ENTRY_ID
MAX_ACTIVE_ENTRIES = _schema.MAX_ACTIVE_ENTRIES
MAX_BYTES = _schema.MAX_BYTES
OVERRIDE_MARKER = _schema.OVERRIDE_MARKER
PAUSED_STATES = _schema.PAUSED_STATES
PLACEHOLDER = _schema.PLACEHOLDER
PORTFOLIO_HEADER = _schema.PORTFOLIO_HEADER
SECRET_PATTERNS = _schema.SECRET_PATTERNS
SENTINEL_ISSUE_VALUES = _schema.SENTINEL_ISSUE_VALUES
SHA_IN_TEXT = _schema.SHA_IN_TEXT
TERMINAL_STATES = _schema.TERMINAL_STATES
VALID_STATES = _schema.VALID_STATES
WARN_ACTIVE_ENTRIES = _schema.WARN_ACTIVE_ENTRIES
WIP_LIMIT_HEADER = _schema.WIP_LIMIT_HEADER
is_implementation_file = _schema.is_implementation_file
parse_entries = _schema.parse_entries
requires_devlog_update = _schema.requires_devlog_update
resolve_canonical_devlog_path = _schema.resolve_canonical_devlog_path
_entry_date = _schema._entry_date

# Re-export validator symbols
_validate_entry = _validator._validate_entry
validate_devlog_content = _validator.validate_devlog_content
validate_repository_devlog = _validator.validate_repository_devlog

__all__ = [
    "ACTIVE_ONLY_FIELDS",
    "ACTIVE_STATES",
    "ARCHIVE_AFTER_DAYS",
    "BASE_FIELDS",
    "BRANCH_STATES",
    "CANONICAL_RELATIVE_PATH",
    "DevLogFinding",
    "ENTRY_HEADING",
    "ENTRY_LOOKUP_HEADING",
    "Entry",
    "FIELD_LINE",
    "ISO_DATE",
    "ISSUE_KEYED_ENTRY_ID",
    "LEADING_DATE",
    "LEGACY_ENTRY_ID",
    "MAX_ACTIVE_ENTRIES",
    "MAX_BYTES",
    "OVERRIDE_MARKER",
    "PAUSED_STATES",
    "PLACEHOLDER",
    "PORTFOLIO_HEADER",
    "SECRET_PATTERNS",
    "SENTINEL_ISSUE_VALUES",
    "SHA_IN_TEXT",
    "TERMINAL_STATES",
    "VALID_STATES",
    "WARN_ACTIVE_ENTRIES",
    "WIP_LIMIT_HEADER",
    "_entry_date",
    "_validate_entry",
    "archive_repository_devlog",
    "archive_terminal_entries",
    "is_implementation_file",
    "main",
    "parse_entries",
    "requires_devlog_update",
    "resolve_canonical_devlog_path",
    "sibling",
    "validate_devlog_content",
    "validate_repository_devlog",
]


def archive_terminal_entries(
    content: str, *, today: date, days: int = ARCHIVE_AFTER_DAYS
) -> tuple[str, dict[int, list[str]]]:
    """Split out shipped/abandoned entries finished more than ``days`` ago.

    Returns the remaining log and the moved entry blocks keyed by the year they
    finished, in document order. Headings and entries without a date stay put.
    """
    assert days >= 0, "days must be non-negative"
    lines = content.splitlines(keepends=True)
    kept: list[str] = []
    archived: dict[int, list[str]] = {}
    index = 0
    while index < len(lines):
        if not ENTRY_HEADING.match(lines[index].rstrip("\r\n")):
            kept.append(lines[index])
            index += 1
            continue
        end = index + 1
        while end < len(lines) and not lines[end].startswith("#"):
            end += 1
        block = "".join(lines[index:end])
        entry = parse_entries(block)[0]
        finished = _entry_date(entry)
        if (
            entry.state in TERMINAL_STATES
            and finished is not None
            and (today - finished).days > days
        ):
            archived.setdefault(finished.year, []).append(block)
        else:
            kept.append(block)
        index = end
    return "".join(kept), archived


def archive_repository_devlog(
    repo_root: Path, *, today: date, days: int = ARCHIVE_AFTER_DAYS
) -> list[Path]:
    """Move old terminal entries into ``DEVELOPMENT_LOG_ARCHIVE_<year>.md``.

    Each run's entries go above earlier runs, so archives read newest first.
    Returns the archive files written.
    """
    log_path = resolve_canonical_devlog_path(repo_root)
    remaining, archived = archive_terminal_entries(
        log_path.read_text(encoding="utf-8"), today=today, days=days
    )
    written: list[Path] = []
    for year, blocks in sorted(archived.items(), reverse=True):
        target = log_path.with_name(f"{log_path.stem}_ARCHIVE_{year}.md")
        header = f"# Development Log Archive — {year}\n\n"
        previous = target.read_text(encoding="utf-8") if target.exists() else header
        body = previous[len(header) :] if previous.startswith(header) else previous
        new = "".join(b if b.endswith("\n\n") else b + "\n" for b in blocks)
        target.write_text(header + new + body, encoding="utf-8", newline="\n")
        written.append(target)
    if archived:
        log_path.write_text(remaining, encoding="utf-8", newline="\n")
    return written


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--changed", nargs="*", default=[])
    parser.add_argument("--warn-only", action="store_true")
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Move shipped/abandoned entries older than --archive-days to the "
        "yearly archive file instead of validating.",
    )
    parser.add_argument("--archive-days", type=int, default=ARCHIVE_AFTER_DAYS)
    parser.add_argument("--today", type=date.fromisoformat, default=None)
    args = parser.parse_args(argv)

    if args.archive:
        written = archive_repository_devlog(
            args.repo_root, today=args.today or date.today(), days=args.archive_days
        )
        for path in written:
            print(f"archived entries into {path}")
        if not written:
            print("Nothing to archive.")
        return 0

    findings = validate_repository_devlog(
        args.repo_root, args.changed, warn_only=args.warn_only
    )
    if not findings:
        print("Development log OK.")
        return 0

    errors = [f for f in findings if f.kind != "portfolio_wip_breach"]
    warnings = [f for f in findings if f.kind == "portfolio_wip_breach"]

    if warnings:
        print("WARNING: development log portfolio WIP breach")
        for finding in warnings:
            loc = f":{finding.line}" if finding.line else ""
            print(f"  - {finding.path.name}{loc} [{finding.kind}]: {finding.message}")
            print(f"      Remediation: {finding.remediation}")

    if errors:
        label = "WARNING" if args.warn_only else "ERROR"
        print(f"{label}: development log")
        for finding in errors:
            loc = f":{finding.line}" if finding.line else ""
            print(f"  - {finding.path.name}{loc} [{finding.kind}]: {finding.message}")
            print(f"      Remediation: {finding.remediation}")
        return 0 if args.warn_only else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
