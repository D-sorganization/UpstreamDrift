#!/usr/bin/env python3
"""Canonical handoff schema validation and enforcement for the repository fleet.

Ensures implementation state survives agent replacement and context exhaustion
by validating canonical handoff schema, detecting unedited placeholders, ensuring
implementation commits update continuation state, and protecting against secrets.

Part of Repository_Management#1938: partitioned into ``handoff_schema.py`` and
``handoff_validator.py``.
Portable: standard library only, copied fleet-wide next to ``development_log.py``.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import re
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


_schema = sibling("handoff_schema")

# Re-export schema symbols
HEX_COMMIT_PATTERN = _schema.HEX_COMMIT_PATTERN
IMPLEMENTATION_PREFIXES = _schema.IMPLEMENTATION_PREFIXES
IMPLEMENTATION_SUFFIXES = _schema.IMPLEMENTATION_SUFFIXES

if TYPE_CHECKING:
    from shared_scripts.handoff_schema import HandoffFinding
else:
    HandoffFinding = _schema.HandoffFinding
OVERRIDE_PATTERN = _schema.OVERRIDE_PATTERN
PLACEHOLDER_PATTERN = _schema.PLACEHOLDER_PATTERN
REQUIRED_HEADINGS = _schema.REQUIRED_HEADINGS
REQUIRED_IDENTITY_FIELDS = _schema.REQUIRED_IDENTITY_FIELDS
SAFE_EXEMPT_PATHS = _schema.SAFE_EXEMPT_PATHS
SAFE_EXEMPT_SUFFIXES = _schema.SAFE_EXEMPT_SUFFIXES
SECRET_PATTERNS = _schema.SECRET_PATTERNS
_archive_policy = sibling("archive_policy")
check_handoff_freshness = _schema.check_handoff_freshness
get_git_head = _schema.get_git_head
is_implementation_file = _schema.is_implementation_file
requires_handoff_update = _schema.requires_handoff_update
resolve_canonical_handoff_path = _schema.resolve_canonical_handoff_path

__all__ = [
    "HEX_COMMIT_PATTERN",
    "HandoffFinding",
    "IMPLEMENTATION_PREFIXES",
    "IMPLEMENTATION_SUFFIXES",
    "OVERRIDE_PATTERN",
    "PLACEHOLDER_PATTERN",
    "REQUIRED_HEADINGS",
    "REQUIRED_IDENTITY_FIELDS",
    "SAFE_EXEMPT_PATHS",
    "SAFE_EXEMPT_SUFFIXES",
    "SECRET_PATTERNS",
    "check_handoff_freshness",
    "get_git_head",
    "is_implementation_file",
    "main",
    "requires_handoff_update",
    "resolve_canonical_handoff_path",
    "sibling",
    "validate_handoff_content",
    "validate_repository_handoff",
]


def _finding(
    path: Path, line: int | None, kind: str, message: str, remediation: str
) -> HandoffFinding:
    """Build one handoff finding (the single construction point for helpers)."""
    return HandoffFinding(
        path=path, line=line, kind=kind, message=message, remediation=remediation
    )


def _check_title(content: str, path: Path) -> list[HandoffFinding]:
    """Level 1 title check.

    Contract: callers skip this check for archive files (RM#1893).
    """
    if content.startswith("# ") or re.search(r"^#\s+.*Handoff", content, re.MULTILINE):
        return []
    return [
        _finding(
            path,
            1,
            "missing_title",
            "Document must start with '# Implementation Handoff'.",
            "Add '# Implementation Handoff' as the first heading.",
        )
    ]


def _check_required_sections(content: str, path: Path) -> list[HandoffFinding]:
    """One finding per missing required section heading, in schema order."""
    return [
        _finding(
            path,
            None,
            "missing_section",
            f"Missing required section '{heading_title}'.",
            f"Add '{heading_title}' section per docs/templates/HANDOFF.md.",
        )
        for heading_title, pattern in REQUIRED_HEADINGS
        if not pattern.search(content)
    ]


def _check_identity_fields(identity_text: str, path: Path) -> list[HandoffFinding]:
    """One finding per required ``## Identity`` field that is absent."""
    findings: list[HandoffFinding] = []
    for field in REQUIRED_IDENTITY_FIELDS:
        field_re = re.compile(
            rf"^-\s+{re.escape(field)}:", re.MULTILINE | re.IGNORECASE
        )
        if not field_re.search(identity_text):
            findings.append(
                _finding(
                    path,
                    None,
                    "missing_field",
                    f"Missing required Identity field '- {field}:'.",
                    f"Add '- {field}: <value>' under '## Identity'.",
                )
            )
    return findings


def _check_implementation_commit(
    identity_text: str, path: Path
) -> list[HandoffFinding]:
    """The implementation commit must be ``SELF`` or a hex SHA."""
    commit_match = re.search(
        r"^-\s+Implementation commit:\s*([^\n]+)",
        identity_text,
        re.MULTILINE | re.IGNORECASE,
    )
    if not commit_match:
        return []
    commit_val = commit_match.group(1).strip()
    first_token = commit_val.split()[0].strip("`'\",")
    if first_token == "SELF" or HEX_COMMIT_PATTERN.match(first_token):
        return []
    return [
        _finding(
            path,
            None,
            "invalid_commit",
            f"Implementation commit '{first_token}' is not 'SELF' or a valid SHA.",
            "Set 'Implementation commit: `SELF`' or the exact SHA.",
        )
    ]


def _check_identity(
    content: str, path: Path, is_template: bool
) -> list[HandoffFinding]:
    """Required identity fields, then the commit value (skipped for templates).

    Contract: callers skip this check for archive files (RM#1893).
    """
    identity_match = re.search(
        r"^##\s+Identity\s*\n(.*?)(?=\n##|\Z)", content, re.DOTALL | re.MULTILINE
    )
    if not identity_match:
        return []
    identity_text = identity_match.group(1)
    findings = _check_identity_fields(identity_text, path)
    if not is_template:
        findings.extend(_check_implementation_commit(identity_text, path))
    return findings


def _check_placeholders(lines: list[str], path: Path) -> list[HandoffFinding]:
    """Unedited ``<...>`` template placeholders outside comments and fences.

    Contract: callers skip this check for templates and archive files.
    """
    findings: list[HandoffFinding] = []
    for idx, line in enumerate(lines, start=1):
        if line.strip().startswith("<!--") or line.strip().startswith("```"):
            continue
        for match in PLACEHOLDER_PATTERN.finditer(line):
            placeholder = match.group(0)
            findings.append(
                _finding(
                    path,
                    idx,
                    "placeholder",
                    f"Unedited template placeholder: {placeholder}",
                    f"Replace '{placeholder}' on line {idx} with evidence.",
                )
            )
    return findings


def _check_changelog(content: str, path: Path) -> list[HandoffFinding]:
    """The ``## Change log`` section needs at least one bullet entry.

    Contract: callers skip this check for archive files (RM#1893).
    """
    changelog_match = re.search(
        r"^##\s+Change\s*log\s*\n(.*?)(?=\n##|\Z)",
        content,
        re.DOTALL | re.MULTILINE | re.IGNORECASE,
    )
    if not changelog_match:
        return []
    changelog_text = changelog_match.group(1).strip()
    has_bullet = any(
        line.strip().startswith("- ") or line.strip().startswith("* ")
        for line in changelog_text.splitlines()
    )
    if has_bullet:
        return []
    return [
        _finding(
            path,
            None,
            "invalid_changelog",
            "Change log must contain at least one bullet entry.",
            "Add '- `SELF` — <changes>' or "
            "'- `SELF` — No material handoff change — <reason>'.",
        )
    ]


def _check_secrets(lines: list[str], path: Path) -> list[HandoffFinding]:
    """Secret scan; the only check that also applies to archive files."""
    findings: list[HandoffFinding] = []
    for idx, line in enumerate(lines, start=1):
        for pattern, secret_type in SECRET_PATTERNS:
            if pattern.search(line):
                findings.append(
                    _finding(
                        path,
                        idx,
                        "secret_detected",
                        f"Potential {secret_type} detected in handoff.",
                        "Remove credentials, tokens, or secret keys from handoff.",
                    )
                )
    return findings


def validate_handoff_content(
    content: str,
    path: Path,
    is_template: bool = False,
) -> list[HandoffFinding]:
    """Validate handoff content against the canonical schema."""
    findings: list[HandoffFinding] = []
    lines = content.splitlines()
    # Archive files are exempt from the format budgets (RM#1893); only the
    # secret scan below still applies to them.
    if not _archive_policy.is_archive_path(path):
        findings.extend(_check_title(content, path))
        findings.extend(_check_required_sections(content, path))
        findings.extend(_check_identity(content, path, is_template))
        if not is_template:
            findings.extend(_check_placeholders(lines, path))
        findings.extend(_check_changelog(content, path))
    findings.extend(_check_secrets(lines, path))
    return findings


def validate_repository_handoff(
    repo_root: Path,
    changed_files: Sequence[str] | None = None,
    warn_only: bool = False,
) -> list[HandoffFinding]:
    """Validate repository handoff status and compliance."""
    findings: list[HandoffFinding] = []
    canonical_path = resolve_canonical_handoff_path(repo_root)

    # Check existence
    if not canonical_path.is_file():
        rel_disp = (
            canonical_path.relative_to(repo_root).as_posix()
            if repo_root in canonical_path.parents or canonical_path == repo_root
            else canonical_path.as_posix()
        )
        findings.append(
            HandoffFinding(
                path=canonical_path,
                line=None,
                kind="missing_handoff",
                message=f"Canonical handoff file not found at '{rel_disp}'.",
                remediation="Create canonical handoff from docs/templates/HANDOFF.md.",
            )
        )
        return findings

    # Read and validate canonical content
    try:
        content = canonical_path.read_text(encoding="utf-8")
        content_findings = validate_handoff_content(content, path=canonical_path)
        findings.extend(content_findings)
        findings.extend(check_handoff_freshness(repo_root, canonical_path, content))
    except OSError as err:
        findings.append(
            HandoffFinding(
                path=canonical_path,
                line=None,
                kind="read_error",
                message=f"Unable to read canonical handoff: {err}",
                remediation="Ensure file exists and has read permissions.",
            )
        )
        return findings

    for archive_file in _archive_policy.archive_files(repo_root):
        for line, kind, message in _archive_policy.check_archive_file(archive_file):
            findings.append(
                HandoffFinding(
                    path=archive_file,
                    line=line,
                    kind=kind,
                    message=message,
                    remediation="Archive files must be valid UTF-8 Markdown.",
                )
            )
        if archive_file.suffix.lower() == ".md":
            findings.extend(
                validate_handoff_content(
                    archive_file.read_text(encoding="utf-8", errors="replace"),
                    path=archive_file,
                )
            )

    # Check commit-level enforcement if changed_files are supplied
    if changed_files:
        norm_changed = [p.replace("\\", "/") for p in changed_files]
        if requires_handoff_update(norm_changed):
            canonical_rel = canonical_path.relative_to(repo_root).as_posix()
            if canonical_rel not in norm_changed and canonical_path.name not in [
                Path(p).name for p in norm_changed
            ]:
                findings.append(
                    HandoffFinding(
                        path=canonical_path,
                        line=None,
                        kind="uncommitted_handoff",
                        message=(
                            "Implementation files modified without updating "
                            f"canonical handoff at '{canonical_rel}'."
                        ),
                        remediation=(
                            f"Stage an update to '{canonical_rel}', or add "
                            "'- `SELF` — No material handoff change — <reason>'."
                        ),
                    )
                )

    return findings


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for handoff validation."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="Root path of repository to validate (default: current directory)",
    )
    parser.add_argument(
        "--warn-only",
        action="store_true",
        help="Report findings as warnings without returning non-zero exit status",
    )
    parser.add_argument(
        "--check-template",
        action="store_true",
        help="Validate docs/templates/HANDOFF.md as a template schema definition",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Optional list of changed files to check for commit enforcement",
    )

    args = parser.parse_args(argv)
    repo_root = args.repo.resolve()

    if args.check_template:
        template_path = repo_root / "docs" / "templates" / "HANDOFF.md"
        if not template_path.is_file():
            print(f"ERROR: Template file not found at {template_path}")
            return 1
        content = template_path.read_text(encoding="utf-8")
        findings = validate_handoff_content(
            content, path=template_path, is_template=True
        )
    else:
        findings = validate_repository_handoff(
            repo_root=repo_root,
            changed_files=args.files or None,
            warn_only=args.warn_only,
        )

    if not findings:
        print("Canonical handoff validation passed.")
        return 0

    label = "WARNING" if args.warn_only else "ERROR"
    print(f"{label}: durable implementation handoff")
    for finding in findings:
        loc = f":{finding.line}" if finding.line else ""
        rel_path = (
            finding.path.relative_to(repo_root).as_posix()
            if repo_root in finding.path.parents or finding.path == repo_root
            else finding.path.as_posix()
        )
        print(f"  - {rel_path}{loc} [{finding.kind}]: {finding.message}")
        print(f"    Remediation: {finding.remediation}")

    return 0 if args.warn_only else 1


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    raise SystemExit(main())
