#!/usr/bin/env python3
"""Canonical development-log schema definitions, models and parsing.

Part of Repository_Management#1938: split out of ``development_log.py``,
which remains the stable public facade and CLI entry point.
Portable: standard library only, copied fleet-wide next to ``handoff_validator.py``.
"""

from __future__ import annotations

import importlib
import importlib.util
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from types import ModuleType


def sibling(name: str) -> ModuleType:
    """Import a sibling fleet module by package, else by file path.

    The fleet copies these modules side by side into repositories that do not
    expose ``shared_scripts`` as an importable package.
    """
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


_handoff = sibling("handoff_validator")
SECRET_PATTERNS = getattr(_handoff, "SECRET_PATTERNS", ())
is_implementation_file = getattr(_handoff, "is_implementation_file", lambda p: False)

CANONICAL_RELATIVE_PATH = Path("docs") / "development" / "DEVELOPMENT_LOG.md"

OVERRIDE_MARKER = re.compile(
    r"<!--\s*CANONICAL-DEVELOPMENT-LOG:\s*([^\s>]+)\s*-->",
)

# The closed state set. An open set invites narration, which is what turns a
# state table back into a journal.
ACTIVE_STATES = frozenset({"proposed", "in_progress", "in_review"})
TERMINAL_STATES = frozenset({"shipped", "abandoned"})
PAUSED_STATES = frozenset({"parked"})
VALID_STATES = ACTIVE_STATES | TERMINAL_STATES | PAUSED_STATES

PORTFOLIO_HEADER = re.compile(
    r"^-\s+\*\*Portfolio:\*\*\s*[`\"']?(?P<portfolio>[a-zA-Z0-9_\-]+)[`\"']?",
    re.MULTILINE | re.IGNORECASE,
)
WIP_LIMIT_HEADER = re.compile(
    r"^-\s+\*\*WIP limit:\*\*\s*[`\"']?(?P<limit>\d+)[`\"']?",
    re.MULTILINE | re.IGNORECASE,
)

# Entry ids. `DL-#<issue>` is the form for every new entry
# (Repository_Management#1520): the governing issue number is unique by
# construction, so two concurrent pull requests can never pick the same id and
# never conflict over the next serial. `DL-0001`-style serial ids are the
# pre-#1520 form and stay valid so existing entries need no rewrite — they are
# already unique — but new ones must not use it.
# The id and title are separated by a middle dot or an en/em dash; fleet logs
# use all of them (Runner_Dashboard's canonical log uses an em dash). A plain
# hyphen is deliberately not accepted: RM's own log has hyphen-headed entries
# that were never validated and would surface unrelated findings.
ENTRY_HEADING = re.compile(
    r"^###\s+(DL-(?:#\d+|\d{4}))\s+[·–—]\s+(.+?)\s*$", re.MULTILINE
)
# Lookup only, never validation: the change-fragment collator also finds a
# legacy ``DL-#N - Title`` entry (UpstreamDrift's log has them), so it updates
# that entry in place instead of inserting a duplicate, and rewrites its
# heading with the middle dot (Repository_Management#1976).
ENTRY_LOOKUP_HEADING = re.compile(
    r"^###\s+(DL-(?:#\d+|\d{4}))\s+[·–—-]\s+(.+?)\s*$", re.MULTILINE
)
LEGACY_ENTRY_ID = re.compile(r"^DL-\d{4}$")
ISSUE_KEYED_ENTRY_ID = re.compile(r"^DL-#(\d+)$")
FIELD_LINE = re.compile(r"^-\s+\*\*(?P<key>[A-Za-z ]+?):\*\*\s*(?P<value>.*?)\s*$")
PLACEHOLDER = re.compile(r"<[^>\n]+>")
SHA_IN_TEXT = re.compile(r"\b[0-9a-fA-F]{7,40}\b")
ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")

# Fields every entry carries, regardless of state.
BASE_FIELDS = ("State", "Owner", "PR", "Paths", "Started", "Last verified", "Summary")
# Additional fields required while an entry is still live.
ACTIVE_ONLY_FIELDS = ("Issue", "Next step")
# Branch only makes sense once code exists.
BRANCH_STATES = frozenset({"in_progress", "in_review"})

# A live entry must name a real governing issue. These placeholders satisfy a
# non-empty check while leaving the entry orphaned by the fleet's own
# definition, so they are rejected for entries in an active state.
SENTINEL_ISSUE_VALUES = frozenset(
    {
        "-",
        "n/a",
        "none",
        "not applicable",
        "not created",
        "tbd",
        "todo",
        "unknown",
    }
)

WARN_ACTIVE_ENTRIES = 30
# Owner limits (Repository_Management#1785): the archiver keeps logs small, so
# the size cap is a backstop rather than a routine blocker.
MAX_ACTIVE_ENTRIES = 75
MAX_BYTES = 200_000
ARCHIVE_AFTER_DAYS = 14
LEADING_DATE = re.compile(r"^\s*`?(\d{4}-\d{2}-\d{2})")


@dataclass(frozen=True)
class DevLogFinding:
    """A single governance finding against a development log."""

    path: Path
    line: int | None
    kind: str
    message: str
    remediation: str


@dataclass
class Entry:
    """One parsed development-log entry."""

    entry_id: str
    title: str
    line: int
    fields: dict[str, str] = field(default_factory=dict)

    @property
    def state(self) -> str:
        """Normalised state string, empty when absent."""
        raw = self.fields.get("State", "").strip()
        if not raw:
            return ""
        return raw.split()[0].strip("`").lower()

    @property
    def is_active(self) -> bool:
        """True when the entry still represents live work."""
        return self.state in ACTIVE_STATES


def resolve_canonical_devlog_path(repo_root: Path) -> Path:
    """Resolve the canonical development-log path for a repository."""
    agents_path = repo_root / "AGENTS.md"
    if agents_path.is_file():
        try:
            text = agents_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            text = ""
        marker = OVERRIDE_MARKER.search(text)
        if marker:
            return repo_root / marker.group(1).strip()
    return repo_root / CANONICAL_RELATIVE_PATH


def requires_devlog_update(changed_paths: Iterable[str]) -> bool:
    """True when any changed path is an implementation file."""
    return any(is_implementation_file(path) for path in changed_paths)


def parse_entries(content: str) -> list[Entry]:
    """Parse development-log entries in document order."""
    lines = content.splitlines()
    entries: list[Entry] = []
    current: Entry | None = None
    for index, raw in enumerate(lines, start=1):
        heading = ENTRY_HEADING.match(raw)
        if heading:
            current = Entry(
                entry_id=heading.group(1),
                title=heading.group(2).strip(),
                line=index,
            )
            entries.append(current)
            continue
        if current is None:
            continue
        if raw.startswith("#"):
            current = None
            continue
        match = FIELD_LINE.match(raw)
        if match:
            current.fields[match.group("key").strip()] = match.group("value").strip()
    return entries


def _entry_date(entry: Entry) -> date | None:
    """The date an entry finished: `Shipped` when present, else `Last verified`."""
    for key in ("Shipped", "Last verified"):
        match = LEADING_DATE.match(entry.fields.get(key, ""))
        if match:
            try:
                return date.fromisoformat(match.group(1))
            except ValueError:
                return None
    return None
