#!/usr/bin/env python3
"""Canonical handoff schema validation and enforcement for the repository fleet.

Ensures implementation state survives agent replacement and context exhaustion
by validating canonical handoff schema, detecting unedited placeholders, ensuring
implementation commits update continuation state, and protecting against secrets.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

REQUIRED_HEADINGS = (
    ("## Identity", re.compile(r"^##\s+Identity\s*$", re.MULTILINE)),
    (
        "## Objective and status",
        re.compile(
            r"^##\s+Objective\s+and\s+status\s*$",
            re.MULTILINE | re.IGNORECASE,
        ),
    ),
    (
        "## Files and decisions",
        re.compile(
            r"^##\s+Files\s+and\s+decisions\s*$",
            re.MULTILINE | re.IGNORECASE,
        ),
    ),
    ("## Validation", re.compile(r"^##\s+Validation\s*$", re.MULTILINE)),
    (
        "## Blockers and risks",
        re.compile(
            r"^##\s+Blockers\s+and\s+risks\s*$",
            re.MULTILINE | re.IGNORECASE,
        ),
    ),
    (
        "## Next steps",
        re.compile(
            r"^##\s+Next\s+steps\s*$",
            re.MULTILINE | re.IGNORECASE,
        ),
    ),
    (
        "## Change log",
        re.compile(
            r"^##\s+Change\s*log\s*$",
            re.MULTILINE | re.IGNORECASE,
        ),
    ),
)

REQUIRED_IDENTITY_FIELDS = (
    "Repository",
    "Working directory",
    "Branch",
    "Baseline commit",
    "Implementation commit",
    "Pull request",
    "Governing issue/epic",
)

PLACEHOLDER_PATTERN = re.compile(r"<[^>\n]+>")
HEX_COMMIT_PATTERN = re.compile(r"^[0-9a-fA-F]{7,40}$")

SECRET_PATTERNS = (
    (re.compile(r"(?:ghp|gho|ghu|ghs|ghr)_[a-zA-Z0-9]{36}"), "GitHub token"),
    (re.compile(r"github_pat_[a-zA-Z0-9_]{82}"), "GitHub fine-grained PAT"),
    (re.compile(r"\bsk-[a-zA-Z0-9]{20,}\b"), "API secret key"),
    (
        re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
        "Private cryptographic key",
    ),
)

IMPLEMENTATION_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".go",
    ".h",
    ".hpp",
    ".js",
    ".jsx",
    ".m",
    ".ps1",
    ".py",
    ".rs",
    ".sh",
    ".ts",
    ".tsx",
    ".yaml",
    ".yml",
    ".toml",
}

IMPLEMENTATION_PREFIXES = (
    "src/",
    "app/",
    "backend/",
    "frontend/",
    "scripts/",
    "shared_scripts/",
    "conductor/",
    "forgejo/",
    ".github/workflows/",
    "tests/",
)

SAFE_EXEMPT_SUFFIXES = {
    ".md",
    ".rst",
    ".txt",
    ".lock",
    ".json",
    ".log",
    ".tmp",
    ".bak",
    ".svg",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
}

SAFE_EXEMPT_PATHS = {
    ".gitignore",
    ".gitattributes",
    ".claudeignore",
    ".prettierignore",
    "LICENSE",
    "SPEC.md",
    "AGENTS.md",
    "CLAUDE.md",
    "AGENT_HANDOFF.md",
    "requirements-lock.txt",
}

OVERRIDE_PATTERN = re.compile(
    r"Canonical handoff(?:\s+location)?\s+is\s+[`\"']?([a-zA-Z0-9_\-./\\]+\.md)[`\"']?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class HandoffFinding:
    """A single governance finding against a handoff document or repository."""

    path: Path
    line: int | None
    kind: str
    message: str
    remediation: str


def resolve_canonical_handoff_path(repo_root: Path) -> Path:
    """Resolve canonical handoff path, defaulting to docs/development/HANDOFF.md."""
    agents_path = repo_root / "AGENTS.md"
    if agents_path.is_file():
        try:
            agents_text = agents_path.read_text(encoding="utf-8", errors="ignore")
            explicit_marker = re.search(
                r"<!--\s*CANONICAL-HANDOFF:\s*([^\s>]+)\s*-->",
                agents_text,
            )
            if explicit_marker:
                override_rel = explicit_marker.group(1).strip()
                return repo_root / override_rel

            override_match = OVERRIDE_PATTERN.search(agents_text)
            if override_match:
                override_rel = override_match.group(1).strip()
                return repo_root / override_rel
        except OSError:
            pass

    return repo_root / "docs" / "development" / "HANDOFF.md"


def is_implementation_file(path_str: str) -> bool:
    """Return True if path_str is a source, workflow, or configuration file."""
    posix = path_str.replace("\\", "/").strip()
    if not posix:
        return False

    name = Path(posix).name
    if name in SAFE_EXEMPT_PATHS:
        return False
    if name == "HANDOFF.md" or posix.endswith("/HANDOFF.md"):
        return False

    if any(
        posix.startswith(prefix)
        for prefix in (
            ".codemap/",
            ".jules/",
            "docs/",
            "reports/",
            "archive/",
            "node_modules/",
            ".venv/",
            "venv/",
        )
    ):
        return False

    suffix = Path(posix).suffix.lower()
    if suffix in SAFE_EXEMPT_SUFFIXES:
        return False

    if suffix in IMPLEMENTATION_SUFFIXES:
        return True

    return any(posix.startswith(prefix) for prefix in IMPLEMENTATION_PREFIXES)


def requires_handoff_update(changed_paths: Iterable[str]) -> bool:
    """Return True if any changed path is an implementation file."""
    return any(is_implementation_file(path) for path in changed_paths)


def validate_handoff_content(
    content: str,
    path: Path,
    is_template: bool = False,
) -> list[HandoffFinding]:
    """Validate handoff content against the canonical schema."""
    findings: list[HandoffFinding] = []
    lines = content.splitlines()

    # 1. Level 1 title check
    if not content.startswith("# ") and not re.search(
        r"^#\s+.*Handoff", content, re.MULTILINE
    ):
        findings.append(
            HandoffFinding(
                path=path,
                line=1,
                kind="missing_title",
                message="Document must start with '# Implementation Handoff'.",
                remediation="Add '# Implementation Handoff' as the first heading.",
            )
        )

    # 2. Required section headings
    for heading_title, pattern in REQUIRED_HEADINGS:
        if not pattern.search(content):
            findings.append(
                HandoffFinding(
                    path=path,
                    line=None,
                    kind="missing_section",
                    message=f"Missing required section '{heading_title}'.",
                    remediation=(
                        f"Add '{heading_title}' section per docs/templates/HANDOFF.md."
                    ),
                )
            )

    # 3. Required identity fields under ## Identity
    identity_match = re.search(
        r"^##\s+Identity\s*\n(.*?)(?=\n##|\Z)", content, re.DOTALL | re.MULTILINE
    )
    if identity_match:
        identity_text = identity_match.group(1)
        for field in REQUIRED_IDENTITY_FIELDS:
            field_re = re.compile(
                rf"^-\s+{re.escape(field)}:", re.MULTILINE | re.IGNORECASE
            )
            if not field_re.search(identity_text):
                findings.append(
                    HandoffFinding(
                        path=path,
                        line=None,
                        kind="missing_field",
                        message=f"Missing required Identity field '- {field}:'.",
                        remediation=f"Add '- {field}: <value>' under '## Identity'.",
                    )
                )

        # Validate Implementation commit value
        commit_match = re.search(
            r"^-\s+Implementation commit:\s*([^\n]+)",
            identity_text,
            re.MULTILINE | re.IGNORECASE,
        )
        if commit_match and not is_template:
            commit_val = commit_match.group(1).strip()
            first_token = commit_val.split()[0].strip("`'\",")
            if first_token != "SELF" and not HEX_COMMIT_PATTERN.match(first_token):
                findings.append(
                    HandoffFinding(
                        path=path,
                        line=None,
                        kind="invalid_commit",
                        message=(
                            f"Implementation commit '{first_token}' is not "
                            "'SELF' or a valid SHA."
                        ),
                        remediation=(
                            "Set 'Implementation commit: `SELF`' or the exact SHA."
                        ),
                    )
                )

    # 4. Check for unedited placeholders when not validating the template itself
    if not is_template:
        for idx, line in enumerate(lines, start=1):
            if line.strip().startswith("<!--") or line.strip().startswith("```"):
                continue
            for match in PLACEHOLDER_PATTERN.finditer(line):
                placeholder = match.group(0)
                findings.append(
                    HandoffFinding(
                        path=path,
                        line=idx,
                        kind="placeholder",
                        message=f"Unedited template placeholder: {placeholder}",
                        remediation=(
                            f"Replace '{placeholder}' on line {idx} with evidence."
                        ),
                    )
                )

    # 5. Validate Change log section
    changelog_match = re.search(
        r"^##\s+Change\s*log\s*\n(.*?)(?=\n##|\Z)",
        content,
        re.DOTALL | re.MULTILINE | re.IGNORECASE,
    )
    if changelog_match:
        changelog_text = changelog_match.group(1).strip()
        bullet_items = [
            line.strip()
            for line in changelog_text.splitlines()
            if line.strip().startswith("- ") or line.strip().startswith("* ")
        ]
        if not bullet_items:
            findings.append(
                HandoffFinding(
                    path=path,
                    line=None,
                    kind="invalid_changelog",
                    message="Change log must contain at least one bullet entry.",
                    remediation=(
                        "Add '- `SELF` — <changes>' or "
                        "'- `SELF` — No material handoff change — <reason>'."
                    ),
                )
            )

    # 6. Secret scanning
    for idx, line in enumerate(lines, start=1):
        for pattern, secret_type in SECRET_PATTERNS:
            if pattern.search(line):
                findings.append(
                    HandoffFinding(
                        path=path,
                        line=idx,
                        kind="secret_detected",
                        message=f"Potential {secret_type} detected in handoff.",
                        remediation=(
                            "Remove credentials, tokens, or secret keys from handoff."
                        ),
                    )
                )

    return findings


def get_git_head(repo_root: Path) -> str | None:
    """Resolve git HEAD commit SHA for the given repository root, if available."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def check_handoff_freshness(
    repo_root: Path,
    handoff_path: Path,
    content: str,
    head_commit: str | None = None,
) -> list[HandoffFinding]:
    """O7: check that HANDOFF.md Implementation commit is not behind HEAD.

    When Implementation commit is 'SELF', it represents in-progress work and passes.
    When it is a SHA, it must match HEAD in a git repository. If git is unavailable
    or the directory is not a git repository, the check passes gracefully.
    """
    commit_match = re.search(
        r"^-\s+Implementation commit:\s*([^\n]+)",
        content,
        re.MULTILINE | re.IGNORECASE,
    )
    if not commit_match:
        return []

    raw_val = commit_match.group(1).strip()
    first_token = raw_val.split()[0].strip("`'\",")
    if first_token == "SELF":
        return []
    if not HEX_COMMIT_PATTERN.match(first_token):
        return []

    head = head_commit if head_commit is not None else get_git_head(repo_root)
    if not head:
        return []

    norm_head = head.strip().lower()
    norm_tok = first_token.lower()
    if norm_head.startswith(norm_tok) or norm_tok.startswith(norm_head):
        return []

    line_number = content[: commit_match.start()].count("\n") + 1
    return [
        HandoffFinding(
            path=handoff_path,
            line=line_number,
            kind="stale_turnover",
            message=(
                f"HANDOFF.md Implementation commit '{first_token}' is "
                f"behind HEAD '{head[:10]}'."
            ),
            remediation=(
                "Update 'Implementation commit' to 'SELF' (for in-progress work) "
                "or to the current HEAD commit SHA."
            ),
        )
    ]


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
