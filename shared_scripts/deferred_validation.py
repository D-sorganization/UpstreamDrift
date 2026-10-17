#!/usr/bin/env python3
"""Validate a repository's deferred external-validation catalog.

Some accepted work cannot be finished by any coding agent: it ends in a
measurement, a laboratory booking, a field collection or a human trial. Left in
the issue queue it is noise that every sweep re-reads and no sweep can act on.
Deleted, its scientific obligation is lost.

The catalog is the third option. `docs/planning/deferred-validation.json` holds
one draft planning record per deferred task — the original issue URL, the
acceptance criteria as written, why the work is external, what resources it
needs, and what must become true to reopen it — so the Board can act on it
later and nothing is silently dropped.

The safeguards enforced here are the ones that make the deferral honest:

* a record can never report validation as done (`validation_state` has no
  "complete"), so deferring is never a way to appear finished;
* evidence must be a checkable reference, never prose, so no measurement is
  conjured into the record;
* the original issue may only be marked closed **after** the record is
  published at a durable URL and a named reviewer signed it off, so discovery
  by keyword never becomes authority to close;
* a `split` disposition must name the implementable remainder that stays open.

This module is portable: it is copied fleet-wide alongside
`handoff_validator.py` and `development_log.py` and must not import anything
outside the standard library.

CLI::

    python -m shared_scripts.deferred_validation --repo-root .

Exit status is 1 when the catalog is present and invalid, 0 when it is valid or
absent — a repository with nothing deferred has nothing to validate.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _load_sibling(name: str) -> Any:
    """Load a shipped sibling without relying on the caller's working directory."""
    import importlib.util

    sibling = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"_fleet_{name}", sibling)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load required catalog dependency: {sibling}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SECRET_PATTERNS = _load_sibling("handoff_validator").SECRET_PATTERNS
CANONICAL_RELATIVE_PATH = Path("docs") / "planning" / "deferred-validation.json"
PUBLISHED_RELATIVE_PATH = Path("docs/development/planning/catalog.json")
SCHEMA_RELATIVE_PATH = Path("docs") / "planning" / "deferred-validation.schema.json"

SCHEMA_VERSION = 1

REQUIRED_TOP_LEVEL_FIELDS: tuple[str, ...] = (
    "schema_version",
    "repository",
    "updated",
    "records",
)

# Record ids are keyed by the governing issue, exactly like development-log
# entry ids (Repository_Management#1520): an issue number is unique by
# construction, so concurrent pull requests never pick the same id.
RECORD_ID = re.compile(r"^DV-#\d+$")
ISSUE_URL = re.compile(
    r"^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/issues/\d+$"
)
HTTPS_URL = re.compile(r"^https://\S+$")
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
REPO_PATH = re.compile(r"^[A-Za-z0-9_.][A-Za-z0-9_./-]*\.[A-Za-z0-9_]+(:\d+)?$")

REQUIRED_RECORD_FIELDS: tuple[str, ...] = (
    "id",
    "title",
    "origin_issue_url",
    "origin_state",
    "disposition",
    "retained_issue_url",
    "blocked_on",
    "validation_state",
    "justification",
    "evidence",
    "resource_prerequisites",
    "acceptance_criteria",
    "reopen_criteria",
    "discovery",
    "reviewed_by",
    "reviewed_on",
    "record_url",
)
OPTIONAL_RECORD_FIELDS: tuple[str, ...] = ("notes",)

DISPOSITIONS = frozenset({"migrate", "split", "retain"})
ORIGIN_STATES = frozenset({"open", "closed_deferred"})
# Deliberately closed and deliberately without a "complete" member.
VALIDATION_STATES = frozenset({"not_started", "blocked_external"})
DISCOVERY_MODES = frozenset({"keyword_match", "manual_review", "board_referral"})
EXTERNAL_BLOCKERS = frozenset(
    {
        "physical_measurement",
        "laboratory_access",
        "field_data_collection",
        "specialized_hardware",
        "human_trial",
        "external_funding",
        "third_party_service",
    }
)

# Obligation fields: a deferral that carries none of these has dropped the work
# rather than parked it.
NON_EMPTY_LIST_FIELDS: tuple[str, ...] = (
    "blocked_on",
    "evidence",
    "resource_prerequisites",
    "acceptance_criteria",
    "reopen_criteria",
)
MIN_JUSTIFICATION_CHARS = 40


@dataclass(frozen=True)
class CatalogFinding:
    """A single governance finding against a deferred-validation catalog."""

    path: Path
    pointer: str
    kind: str
    message: str
    remediation: str


def _finding(
    path: Path, pointer: str, kind: str, message: str, remediation: str
) -> CatalogFinding:
    return CatalogFinding(
        path=path,
        pointer=pointer,
        kind=kind,
        message=message,
        remediation=remediation,
    )


def _is_checkable_reference(value: str) -> bool:
    """True when evidence points at something a reader can open."""
    text = value.strip()
    return bool(HTTPS_URL.match(text) or REPO_PATH.match(text))


def _validate_enum(
    record: dict[str, Any],
    field: str,
    allowed: frozenset[str],
    path: Path,
    pointer: str,
) -> list[CatalogFinding]:
    value = record.get(field)
    if isinstance(value, str) and value in allowed:
        return []
    return [
        _finding(
            path,
            f"{pointer}/{field}",
            "bad_value",
            f"{field} is {value!r}; allowed values are {', '.join(sorted(allowed))}.",
            f"Set {field} to one of the allowed values.",
        )
    ]


def _check_field_presence(
    record: dict[str, Any], path: Path, pointer: str
) -> list[CatalogFinding]:
    """Unknown fields first (sorted), then missing required fields.

    Contract: ``record`` is a mapping; a non-empty result means every later
    check must be skipped because required keys may be absent.
    """
    findings: list[CatalogFinding] = []
    known = set(REQUIRED_RECORD_FIELDS) | set(OPTIONAL_RECORD_FIELDS)
    for field in sorted(set(record) - known):
        findings.append(
            _finding(
                path,
                f"{pointer}/{field}",
                "unknown_field",
                f"Unknown record field {field!r}.",
                "Remove the field; the catalog schema is closed. Measurements "
                "belong in the repository's results, never in a deferral record.",
            )
        )
    for field in REQUIRED_RECORD_FIELDS:
        if field not in record:
            findings.append(
                _finding(
                    path,
                    f"{pointer}/{field}",
                    "missing_field",
                    f"Required record field {field!r} is missing.",
                    f"Add {field}; write null where the schema allows it.",
                )
            )
    return findings


def _check_identity(
    record: dict[str, Any], path: Path, pointer: str
) -> list[CatalogFinding]:
    """Record id, title and origin issue URL.

    Contract: every key in ``REQUIRED_RECORD_FIELDS`` is present in ``record``.
    """
    findings: list[CatalogFinding] = []
    record_id = record["id"]
    if not (isinstance(record_id, str) and RECORD_ID.match(record_id)):
        findings.append(
            _finding(
                path,
                f"{pointer}/id",
                "bad_id",
                f"Record id {record_id!r} is not of the form DV-#<issue>.",
                "Key the record by its governing issue number, never a serial.",
            )
        )
    if not (isinstance(record["title"], str) and record["title"].strip()):
        findings.append(
            _finding(
                path,
                f"{pointer}/title",
                "empty_field",
                "title is empty.",
                "Give the deferred task a one-line title.",
            )
        )
    origin = record["origin_issue_url"]
    if not (isinstance(origin, str) and ISSUE_URL.match(origin)):
        findings.append(
            _finding(
                path,
                f"{pointer}/origin_issue_url",
                "bad_value",
                f"origin_issue_url {origin!r} is not a GitHub issue URL.",
                "Preserve the original issue URL in full; it is the audit trail.",
            )
        )
    return findings


def _check_enums(
    record: dict[str, Any], path: Path, pointer: str
) -> list[CatalogFinding]:
    """The four closed value sets, in schema order."""
    findings: list[CatalogFinding] = []
    for field, allowed in (
        ("disposition", DISPOSITIONS),
        ("origin_state", ORIGIN_STATES),
        ("validation_state", VALIDATION_STATES),
        ("discovery", DISCOVERY_MODES),
    ):
        findings.extend(_validate_enum(record, field, allowed, path, pointer))
    return findings


def _check_obligation_lists(
    record: dict[str, Any], path: Path, pointer: str
) -> list[CatalogFinding]:
    """Each obligation field must be a non-empty list of non-empty strings."""
    findings: list[CatalogFinding] = []
    for field in NON_EMPTY_LIST_FIELDS:
        value = record[field]
        if not isinstance(value, list) or not value:
            findings.append(
                _finding(
                    path,
                    f"{pointer}/{field}",
                    "empty_field",
                    f"{field} is empty.",
                    "The obligation survives the deferral: record what the "
                    "original issue demanded and what would bring it back.",
                )
            )
            continue
        for index, item in enumerate(value):
            if not isinstance(item, str) or not item.strip():
                findings.append(
                    _finding(
                        path,
                        f"{pointer}/{field}/{index}",
                        "empty_field",
                        f"{field}[{index}] is not a non-empty string.",
                        f"Write each {field} entry as text.",
                    )
                )
    return findings


def _check_blockers(
    record: dict[str, Any], path: Path, pointer: str
) -> list[CatalogFinding]:
    """Every blocker must name a physical-world reason."""
    findings: list[CatalogFinding] = []
    blockers = record["blocked_on"]
    if not isinstance(blockers, list):
        return findings
    for index, item in enumerate(blockers):
        if isinstance(item, str) and item in EXTERNAL_BLOCKERS:
            continue
        findings.append(
            _finding(
                path,
                f"{pointer}/blocked_on/{index}",
                "bad_value",
                f"blocked_on[{index}] is {item!r}; allowed values are "
                f"{', '.join(sorted(EXTERNAL_BLOCKERS))}.",
                "A deferral is for work blocked on the physical world, not "
                "for work that is merely hard or unscheduled.",
            )
        )
    return findings


def _check_evidence(
    record: dict[str, Any], path: Path, pointer: str
) -> list[CatalogFinding]:
    """Evidence entries must be URLs or repo-relative paths, never prose."""
    findings: list[CatalogFinding] = []
    evidence = record["evidence"]
    if not isinstance(evidence, list):
        return findings
    for index, item in enumerate(evidence):
        if isinstance(item, str) and _is_checkable_reference(item):
            continue
        findings.append(
            _finding(
                path,
                f"{pointer}/evidence/{index}",
                "unverifiable_evidence",
                f"evidence[{index}] is not a URL or repo-relative path.",
                "Cite something a reader can open. Prose is not evidence, "
                "and a measurement that was never taken is not evidence.",
            )
        )
    return findings


def _check_justification(
    record: dict[str, Any], path: Path, pointer: str
) -> list[CatalogFinding]:
    """The justification must be long enough to carry a real reason."""
    justification = record["justification"]
    if (
        isinstance(justification, str)
        and len(justification.strip()) >= MIN_JUSTIFICATION_CHARS
    ):
        return []
    return [
        _finding(
            path,
            f"{pointer}/justification",
            "thin_justification",
            f"justification is shorter than {MIN_JUSTIFICATION_CHARS} characters.",
            "State why no coding agent can finish this work.",
        )
    ]


def _check_dates_and_urls(
    record: dict[str, Any], path: Path, pointer: str
) -> list[CatalogFinding]:
    """``reviewed_on`` is an ISO date or null; the URL fields are URLs or null."""
    findings: list[CatalogFinding] = []
    reviewed_on = record["reviewed_on"]
    if reviewed_on is not None and not (
        isinstance(reviewed_on, str) and ISO_DATE.match(reviewed_on)
    ):
        findings.append(
            _finding(
                path,
                f"{pointer}/reviewed_on",
                "bad_value",
                f"reviewed_on {reviewed_on!r} is not an ISO date (YYYY-MM-DD) or null.",
                "Write reviewed_on as YYYY-MM-DD.",
            )
        )
    for field, pattern in (
        ("retained_issue_url", ISSUE_URL),
        ("record_url", HTTPS_URL),
    ):
        value = record[field]
        if value is not None and not (isinstance(value, str) and pattern.match(value)):
            findings.append(
                _finding(
                    path,
                    f"{pointer}/{field}",
                    "bad_value",
                    f"{field} {value!r} is not a valid URL or null.",
                    f"Write {field} as a full https URL, or null.",
                )
            )
    return findings


def _validate_record_shape(
    record: dict[str, Any], path: Path, pointer: str
) -> list[CatalogFinding]:
    """Required fields, unknown fields, and closed value sets."""
    findings = _check_field_presence(record, path, pointer)
    if findings:
        return findings
    for check in (
        _check_identity,
        _check_enums,
        _check_obligation_lists,
        _check_blockers,
        _check_evidence,
        _check_justification,
        _check_dates_and_urls,
    ):
        findings.extend(check(record, path, pointer))
    return findings


def _validate_record_safeguards(
    record: dict[str, Any], path: Path, pointer: str
) -> list[CatalogFinding]:
    """Publish-then-close ordering, split remainders, retained work."""
    findings: list[CatalogFinding] = []
    origin_state = record.get("origin_state")
    disposition = record.get("disposition")

    if origin_state == "closed_deferred":
        if not record.get("record_url"):
            findings.append(
                _finding(
                    path,
                    f"{pointer}/record_url",
                    "closed_without_record",
                    "The original issue is marked closed but this record has no "
                    "durable published link.",
                    "Publish the record, verify the link resolves on the default "
                    "branch, and only then close the original as not planned.",
                )
            )
        if not (record.get("reviewed_by") and record.get("reviewed_on")):
            findings.append(
                _finding(
                    path,
                    f"{pointer}/reviewed_by",
                    "closed_without_review",
                    "The original issue is marked closed but no reviewer and date "
                    "are recorded.",
                    "Keyword matching is candidate discovery, never authority to "
                    "close. Record the human or Board review that approved it.",
                )
            )
        if disposition == "retain":
            findings.append(
                _finding(
                    path,
                    f"{pointer}/disposition",
                    "retained_but_closed",
                    "Disposition is 'retain' but the original issue is marked "
                    "closed as deferred.",
                    "Implementable software and CI work stays in the queue; "
                    "record it as retained and leave the issue open.",
                )
            )

    if disposition == "split" and not record.get("retained_issue_url"):
        findings.append(
            _finding(
                path,
                f"{pointer}/retained_issue_url",
                "split_without_remainder",
                "Disposition is 'split' but no retained issue is named.",
                "Split means the implementable remainder stays open: file or "
                "name it, then link it here.",
            )
        )
    return findings


def _scan_secrets(value: Any, path: Path, pointer: str) -> list[CatalogFinding]:
    """Reject credentials anywhere in the catalog."""
    findings: list[CatalogFinding] = []
    if isinstance(value, str):
        for pattern, secret_type in SECRET_PATTERNS:
            if pattern.search(value):
                findings.append(
                    _finding(
                        path,
                        pointer,
                        "secret_detected",
                        f"Potential {secret_type} detected in the catalog.",
                        "Remove credentials, tokens, and secret keys.",
                    )
                )
    elif isinstance(value, dict):
        for key, item in value.items():
            findings.extend(_scan_secrets(item, path, f"{pointer}/{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            findings.extend(_scan_secrets(item, path, f"{pointer}/{index}"))
    return findings


def _check_top_level_fields(data: dict[str, Any], path: Path) -> list[CatalogFinding]:
    """Unknown top-level fields first (sorted), then missing required fields.

    Contract: ``data`` is the parsed catalog root; a non-empty result means the
    caller must stop, because required keys may be absent.
    """
    findings: list[CatalogFinding] = []
    for field in sorted(set(data) - set(REQUIRED_TOP_LEVEL_FIELDS)):
        findings.append(
            _finding(
                path,
                f"/{field}",
                "unknown_field",
                f"Unknown top-level field {field!r}.",
                "Remove the field; the catalog schema is closed.",
            )
        )
    for field in REQUIRED_TOP_LEVEL_FIELDS:
        if field not in data:
            findings.append(
                _finding(
                    path,
                    f"/{field}",
                    "missing_field",
                    f"Required field {field!r} is missing.",
                    f"Add {field} to the catalog.",
                )
            )
    return findings


def _check_catalog_header(data: dict[str, Any], path: Path) -> list[CatalogFinding]:
    """``schema_version``, ``repository`` and ``updated``.

    Contract: every key in ``REQUIRED_TOP_LEVEL_FIELDS`` is present in ``data``.
    """
    findings: list[CatalogFinding] = []
    if data["schema_version"] != SCHEMA_VERSION:
        findings.append(
            _finding(
                path,
                "/schema_version",
                "bad_value",
                f"schema_version must be {SCHEMA_VERSION}.",
                "Migrate the catalog to the published schema version.",
            )
        )
    if not (isinstance(data["repository"], str) and data["repository"].strip()):
        findings.append(
            _finding(
                path,
                "/repository",
                "empty_field",
                "repository is empty.",
                "Name the repository the catalog belongs to.",
            )
        )
    updated = data["updated"]
    if not (isinstance(updated, str) and ISO_DATE.match(updated)):
        findings.append(
            _finding(
                path,
                "/updated",
                "bad_value",
                f"updated {updated!r} is not an ISO date (YYYY-MM-DD).",
                "Refresh `updated` whenever a record changes.",
            )
        )
    return findings


def _check_record(record: Any, path: Path, pointer: str) -> list[CatalogFinding]:
    """Shape, then (only when the shape is clean) the publish-then-close rules."""
    if not isinstance(record, dict):
        return [
            _finding(
                path,
                pointer,
                "bad_value",
                "Record is not a JSON object.",
                "See docs/planning/deferred-validation.schema.json.",
            )
        ]
    shape = _validate_record_shape(record, path, pointer)
    if shape:
        return shape
    return _validate_record_safeguards(record, path, pointer)


def _check_duplicate_id(
    record: Any, seen: set[str], path: Path, pointer: str
) -> list[CatalogFinding]:
    """Flag a repeated record id and remember every string id in ``seen``.

    Postcondition: ``seen`` contains the record's id when it is a string.
    """
    record_id = record.get("id") if isinstance(record, dict) else None
    if not isinstance(record_id, str):
        return []
    findings: list[CatalogFinding] = []
    if record_id in seen:
        findings.append(
            _finding(
                path,
                f"{pointer}/id",
                "duplicate_id",
                f"Record id {record_id} appears more than once.",
                "One record per deferred task, updated in place.",
            )
        )
    seen.add(record_id)
    return findings


def _check_records(records: list[Any], path: Path) -> list[CatalogFinding]:
    """Validate each record in order, then report duplicate ids per record."""
    findings: list[CatalogFinding] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        pointer = f"/records/{index}"
        findings.extend(_check_record(record, path, pointer))
        findings.extend(_check_duplicate_id(record, seen, path, pointer))
    return findings


def validate_catalog(data: Any, path: Path) -> list[CatalogFinding]:
    """Validate a parsed catalog against the published schema and safeguards."""
    if not isinstance(data, dict):
        return [
            _finding(
                path,
                "",
                "bad_value",
                "The catalog root is not a JSON object.",
                "See docs/planning/deferred-validation.schema.json.",
            )
        ]
    findings = _check_top_level_fields(data, path)
    if findings:
        return findings
    findings.extend(_check_catalog_header(data, path))
    records = data["records"]
    if not isinstance(records, list):
        findings.append(
            _finding(
                path,
                "/records",
                "bad_value",
                "records is not a list.",
                "Write records as a JSON array; an empty array is valid.",
            )
        )
        return findings
    findings.extend(_check_records(records, path))
    findings.extend(_scan_secrets(data, path, ""))
    return findings


def _validate_published_catalog(path: Path) -> list[CatalogFinding]:
    """Reuse the deployed v1 contract, including its local evidence artifacts."""
    try:
        checker = _load_sibling("deferred_planning")
    except (ImportError, OSError) as exc:
        return [
            _finding(
                path,
                "",
                "missing_checker",
                str(exc),
                "Install the complete deferred-validation validator bundle.",
            )
        ]
    try:
        data = checker.validate_catalog(path.parent)
    except (ValueError, OSError, UnicodeError) as exc:
        return [
            _finding(
                path,
                "",
                "invalid_published_catalog",
                str(exc),
                "Restore the v1 catalog and its original plan/source artifacts.",
            )
        ]
    return _scan_secrets(data, path, "")


def validate_repository_catalog(repo_root: Path) -> list[CatalogFinding]:
    """Validate the single authority, including v1 catalogs already published.

    Never choose between two catalogs silently or infer Board review during
    schema recognition. This read-only operation does not migrate or close work.
    """
    path = repo_root / CANONICAL_RELATIVE_PATH
    published = repo_root / PUBLISHED_RELATIVE_PATH
    if (path.exists() or path.is_symlink()) and (
        published.exists() or published.is_symlink()
    ):
        return [
            _finding(
                path,
                "",
                "conflicting_catalogs",
                "Two planning catalogs exist.",
                "Reconcile into one reviewed authority without losing evidence.",
            )
        ]
    if published.exists() or published.is_symlink():
        return _validate_published_catalog(published)
    if not path.exists() and not path.is_symlink():
        return []
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return [
            _finding(
                path,
                "",
                "unreadable",
                f"Cannot read the catalog: {exc}.",
                "Restore the file or remove it.",
            )
        ]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return [
            _finding(
                path,
                "",
                "unparseable",
                f"The catalog is not valid JSON: {exc}.",
                "Fix the JSON syntax; the catalog is machine-read.",
            )
        ]
    return validate_catalog(data, path)


def format_findings(findings: list[CatalogFinding]) -> list[str]:
    """Render findings as one line each."""
    return [
        f"{finding.path.as_posix()}{finding.pointer} [{finding.kind}]: "
        f"{finding.message} (Remediation: {finding.remediation})"
        for finding in findings
    ]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root containing docs/planning/deferred-validation.json.",
    )
    args = parser.parse_args(argv)

    findings = validate_repository_catalog(args.repo_root.resolve())
    if not findings:
        print("Deferred-validation catalog OK.")
        return 0
    print("Deferred-validation catalog failed validation:")
    for line in format_findings(findings):
        print(f"  - {line}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
