#!/usr/bin/env python3
"""Canonical development-log validation logic.

Part of Repository_Management#1938: split out of ``development_log.py``,
which remains the stable public facade and CLI entry point.
Portable: standard library only, copied fleet-wide next to ``handoff_validator.py``.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING


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

ACTIVE_ONLY_FIELDS = _schema.ACTIVE_ONLY_FIELDS
ACTIVE_STATES = _schema.ACTIVE_STATES
BASE_FIELDS = _schema.BASE_FIELDS
BRANCH_STATES = _schema.BRANCH_STATES
ISO_DATE = _schema.ISO_DATE
ISSUE_KEYED_ENTRY_ID = _schema.ISSUE_KEYED_ENTRY_ID
MAX_ACTIVE_ENTRIES = _schema.MAX_ACTIVE_ENTRIES
MAX_BYTES = _schema.MAX_BYTES
PLACEHOLDER = _schema.PLACEHOLDER
PORTFOLIO_HEADER = _schema.PORTFOLIO_HEADER
SECRET_PATTERNS = _schema.SECRET_PATTERNS
SENTINEL_ISSUE_VALUES = _schema.SENTINEL_ISSUE_VALUES
SHA_IN_TEXT = _schema.SHA_IN_TEXT
VALID_STATES = _schema.VALID_STATES
WIP_LIMIT_HEADER = _schema.WIP_LIMIT_HEADER
parse_entries = _schema.parse_entries
requires_devlog_update = _schema.requires_devlog_update
resolve_canonical_devlog_path = _schema.resolve_canonical_devlog_path

if TYPE_CHECKING:
    from shared_scripts.development_log_schema import DevLogFinding, Entry
else:
    DevLogFinding = _schema.DevLogFinding
    Entry = _schema.Entry


def _validate_entry(entry: Entry, path: Path) -> list[DevLogFinding]:
    """Validate a single development-log entry against the canonical schema."""
    findings: list[DevLogFinding] = []

    state = entry.state
    if not state:
        findings.append(
            DevLogFinding(
                path=path,
                line=entry.line,
                kind="missing_state",
                message=f"{entry.entry_id} has no State field.",
                remediation=f"Add '- **State:** <one of {sorted(VALID_STATES)}>'.",
            )
        )
    elif state not in VALID_STATES:
        findings.append(
            DevLogFinding(
                path=path,
                line=entry.line,
                kind="invalid_state",
                message=f"{entry.entry_id} has unknown state '{state}'.",
                remediation=f"Use one of: {', '.join(sorted(VALID_STATES))}.",
            )
        )

    # An issue-keyed id must actually name the entry's governing issue,
    # otherwise the id is unique but meaningless and the entry/issue join the
    # orphan detection depends on silently breaks (Repository_Management#1520).
    issue_keyed = ISSUE_KEYED_ENTRY_ID.match(entry.entry_id)
    if issue_keyed:
        issue_field = entry.fields.get("Issue", "")
        if f"#{issue_keyed.group(1)}" not in issue_field:
            findings.append(
                DevLogFinding(
                    path=path,
                    line=entry.line,
                    kind="entry_id_issue_mismatch",
                    message=(
                        f"{entry.entry_id} is keyed by issue "
                        f"#{issue_keyed.group(1)}, which its Issue field "
                        f"({issue_field or 'empty'}) does not name."
                    ),
                    remediation=(
                        "Key the entry by its governing issue, or correct the "
                        "Issue field."
                    ),
                )
            )

    required = list(BASE_FIELDS)
    if state in ACTIVE_STATES:
        required.extend(ACTIVE_ONLY_FIELDS)
    if state in BRANCH_STATES:
        required.append("Branch")
    if state == "parked":
        required.append("Parked")

    for key in required:
        value = entry.fields.get(key, "").strip()
        if not value:
            findings.append(
                DevLogFinding(
                    path=path,
                    line=entry.line,
                    kind="missing_field",
                    message=f"{entry.entry_id} is missing required field '{key}'.",
                    remediation=f"Add '- **{key}:** ...' to {entry.entry_id}.",
                )
            )
        elif PLACEHOLDER.search(value):
            findings.append(
                DevLogFinding(
                    path=path,
                    line=entry.line,
                    kind="placeholder",
                    message=(f"{entry.entry_id} field '{key}' holds a placeholder."),
                    remediation="Replace angle-bracket placeholders with real values.",
                )
            )

    issue = entry.fields.get("Issue", "").strip()
    if state in ACTIVE_STATES and issue:
        bare = issue.strip("`").split("—")[0].strip().rstrip(".").lower()
        if bare in SENTINEL_ISSUE_VALUES:
            findings.append(
                DevLogFinding(
                    path=path,
                    line=entry.line,
                    kind="sentinel_issue",
                    message=(
                        f"{entry.entry_id} is {state} but its Issue is "
                        f"'{issue}'. A live entry must name a real governing "
                        "issue."
                    ),
                    remediation="Open a governing issue and reference it.",
                )
            )

    verified = entry.fields.get("Last verified", "")
    if verified and not ISO_DATE.search(verified):
        findings.append(
            DevLogFinding(
                path=path,
                line=entry.line,
                kind="unusable_last_verified",
                message=f"{entry.entry_id} 'Last verified' has no YYYY-MM-DD date.",
                remediation="Write 'Last verified' as '<YYYY-MM-DD> (`<sha>`)'.",
            )
        )
    if verified and not SHA_IN_TEXT.search(verified):
        findings.append(
            DevLogFinding(
                path=path,
                line=entry.line,
                kind="unusable_last_verified",
                message=f"{entry.entry_id} 'Last verified' has no commit SHA.",
                remediation="Include the verifying commit SHA in 'Last verified'.",
            )
        )
    return findings


def validate_devlog_content(
    content: str,
    path: Path,
    is_template: bool = False,
    check_portfolio_wip: bool = False,
) -> list[DevLogFinding]:
    """Validate development-log content against the canonical schema."""
    findings: list[DevLogFinding] = []

    if "## Active" not in content:
        findings.append(
            DevLogFinding(
                path=path,
                line=None,
                kind="missing_heading",
                message="Development log has no '## Active' section.",
                remediation="Create the log from docs/templates/DEVELOPMENT_LOG.md.",
            )
        )

    for pattern, secret_type in SECRET_PATTERNS:
        if pattern.search(content):
            findings.append(
                DevLogFinding(
                    path=path,
                    line=None,
                    kind="secret_detected",
                    message=f"Potential {secret_type} detected in development log.",
                    remediation="Remove credentials or tokens from the log.",
                )
            )

    entries = parse_entries(content)
    seen: dict[str, int] = {}
    for entry in entries:
        if entry.entry_id in seen:
            findings.append(
                DevLogFinding(
                    path=path,
                    line=entry.line,
                    kind="duplicate_entry",
                    message=(
                        f"{entry.entry_id} appears twice "
                        f"(first at line {seen[entry.entry_id]}). "
                        "Entries are updated in place, never duplicated."
                    ),
                    remediation="Merge the duplicate into the original entry.",
                )
            )
            continue
        seen[entry.entry_id] = entry.line

    if is_template:
        return findings

    for entry in entries:
        findings.extend(_validate_entry(entry, path))

    active = [e for e in entries if e.is_active]
    if len(active) > MAX_ACTIVE_ENTRIES:
        findings.append(
            DevLogFinding(
                path=path,
                line=None,
                kind="wip_ceiling",
                message=(
                    f"{len(active)} active entries exceeds the ceiling of "
                    f"{MAX_ACTIVE_ENTRIES}. Work is being started faster than "
                    "it is finished."
                ),
                remediation="Park or ship entries before opening new ones.",
            )
        )

    # Check portfolio WIP limit if declared in header and enabled (#1465)
    if check_portfolio_wip:
        wip_match = WIP_LIMIT_HEADER.search(content)
        if wip_match:
            try:
                port_limit = int(wip_match.group("limit"))
                port_match = PORTFOLIO_HEADER.search(content)
                port_name = (
                    port_match.group("portfolio").lower() if port_match else "portfolio"
                )
                # Only in_progress is active effort; in_review waits on CI or
                # merge (Repository_Management#1785).
                concurrent_wip = [e for e in entries if e.state == "in_progress"]
                if port_limit > 0 and len(concurrent_wip) > port_limit:
                    findings.append(
                        DevLogFinding(
                            path=path,
                            line=None,
                            kind="portfolio_wip_breach",
                            message=(
                                f"{len(concurrent_wip)} concurrent WIP entries "
                                f"(in_progress) exceeds the '{port_name}' "
                                f"portfolio cap of {port_limit}."
                            ),
                            remediation=(
                                f"Park or ship entries to bring WIP to <= {port_limit} "
                                "before starting new work."
                            ),
                        )
                    )
            except ValueError:
                pass

    if len(content.encode("utf-8")) > MAX_BYTES:
        findings.append(
            DevLogFinding(
                path=path,
                line=None,
                kind="size_ceiling",
                message=f"Development log exceeds {MAX_BYTES} bytes.",
                remediation=(
                    "Move shipped entries to DEVELOPMENT_LOG_ARCHIVE_<year>.md."
                ),
            )
        )
    return findings


def validate_repository_devlog(
    repo_root: Path,
    changed_files: Sequence[str],
    warn_only: bool = False,
) -> list[DevLogFinding]:
    """Validate a repository's canonical development log.

    When implementation files changed, the log must either be updated in the
    same commit or carry the explicit no-change escape line.
    """
    path = resolve_canonical_devlog_path(repo_root)
    if repo_root in path.parents:
        rel = path.relative_to(repo_root).as_posix()
    else:
        rel = str(path)

    if not path.is_file():
        # warn_only controls the caller's exit status, never whether a finding
        # is reported. Suppressing it here silenced exactly the repositories
        # that still need bootstrapping.
        if not requires_devlog_update(changed_files):
            return []
        return [
            DevLogFinding(
                path=path,
                line=None,
                kind="missing_devlog",
                message=f"No development log at '{rel}'.",
                remediation=(
                    "Create it from docs/templates/DEVELOPMENT_LOG.md, or run "
                    "python -m scripts.bootstrap_development_log"
                ),
            )
        ]

    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        return [
            DevLogFinding(
                path=path,
                line=None,
                kind="unreadable",
                message=f"Could not read '{rel}': {exc}",
                remediation="Ensure the development log is readable.",
            )
        ]

    touched = {Path(p).as_posix() for p in changed_files}
    check_wip = rel in touched or not changed_files
    findings = validate_devlog_content(content, path, check_portfolio_wip=check_wip)

    if requires_devlog_update(changed_files) and rel not in touched:
        findings.append(
            DevLogFinding(
                path=path,
                line=None,
                kind="uncommitted_devlog",
                message=f"Implementation files changed without updating '{rel}'.",
                remediation=(
                    "Refresh the affected entry's 'Last verified' and 'Next "
                    "step'. If nothing material changed, stage the log with "
                    "'No material development-log change — <reason>' recorded in "
                    "it. Staging is what satisfies this check; the presence of "
                    "that phrase from an earlier commit does not."
                ),
            )
        )
    return findings
