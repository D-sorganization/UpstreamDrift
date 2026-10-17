#!/usr/bin/env python3
"""Canonical handoff schema definitions, models, and file classification.

Part of Repository_Management#1938: split out of ``handoff_validator.py``,
which remains the stable public facade and CLI entry point.
Portable: standard library only, copied fleet-wide next to ``development_log.py``.
"""

from __future__ import annotations

import importlib
import importlib.util
import re
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
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
