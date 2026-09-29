#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = [
    ROOT / "docs" / "README.md",
    ROOT / "docs" / "assessments" / "README.md",
    ROOT / "docs" / "adr" / "README.md",
    ROOT / "docs" / "adr" / "ADR_TEMPLATE.md",
    ROOT / "docs" / "governance" / "DOCS_GOVERNANCE.md",
]
CANONICAL_PROCESS_DIRECTORY_NAMES = ("assessments", "issues")
SOURCE_OF_TRUTH_HEADINGS = {
    "SPEC.md": {
        "SPEC Ownership and Update Cadence",
        "Quality Gates",
        "Architecture Principles",
        "Support Matrix",
    }
}
ROOT_ALLOWLIST_CONFIG = ROOT / "scripts" / "config" / "root_allowlist.json"


def _tracked_root_entries() -> list[str] | None:
    """Return tracked root entries via git ls-tree, or None on git command failure."""
    cp = subprocess.run(
        ["git", "ls-tree", "--name-only", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if cp.returncode != 0:
        return None
    return [line.strip() for line in cp.stdout.splitlines() if line.strip()]


def _load_root_allowlist(config_path: Path | None = None) -> list[str]:
    """Load and validate the repository root allowlist config.

    Preconditions:
        - config file exists and contains valid JSON object
        - 'entries' key is present and is a list of strings
        - no entries contain path separators ('/' or '\\')
        - no entries are empty strings
        - no duplicate entries exist
    Postconditions:
        - returns list of distinct root entry names
    """
    path = config_path if config_path is not None else ROOT_ALLOWLIST_CONFIG
    if not path.exists():
        raise ValueError(f"Root allowlist config not found: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in root allowlist config {path}: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(f"Root allowlist config at {path} must be a JSON object")

    if "entries" not in data:
        raise ValueError(f"Root allowlist config at {path} missing 'entries' key")

    raw_entries = data["entries"]
    if not isinstance(raw_entries, list):
        raise ValueError(f"Root allowlist 'entries' in {path} must be a list")

    seen: set[str] = set()
    entries: list[str] = []
    for entry in raw_entries:
        if not isinstance(entry, str):
            raise ValueError(
                f"Allowlist entry must be a string, got {type(entry).__name__}: {entry!r}"
            )
        if "/" in entry or "\\" in entry:
            raise ValueError(
                f"Allowlist entry cannot contain path separators ('/' or '\\'): {entry!r}"
            )
        if not entry.strip():
            raise ValueError(f"Allowlist entry cannot be empty: {entry!r}")
        if entry in seen:
            raise ValueError(f"Duplicate root allowlist entry: {entry!r}")
        seen.add(entry)
        entries.append(entry)

    return entries


def _unexpected_root_entries(
    tracked: Iterable[str], allowlist: Iterable[str]
) -> list[str]:
    """Return tracked root entries not present in the allowlist, sorted."""
    allowlist_set = set(allowlist)
    return sorted(entry for entry in set(tracked) if entry not in allowlist_set)


def _stale_root_entries(tracked: Iterable[str], allowlist: Iterable[str]) -> list[str]:
    """Return allowlist entries that are no longer tracked, sorted."""
    tracked_set = set(tracked)
    return sorted(entry for entry in set(allowlist) if entry not in tracked_set)


def _root_allowlist_failure() -> str | None:
    """Failure message for root entries outside the allowlist or stale entries, else None.

    Fails closed: an invalid config or a failed ``git ls-tree`` is a failure.
    """
    try:
        allowlist = _load_root_allowlist()
    except ValueError as exc:
        return f"Invalid root allowlist config:\n- {exc}"
    tracked_root = _tracked_root_entries()
    if tracked_root is None:
        return "Failed to list tracked repository root entries with git ls-tree"
    unexpected_root = _unexpected_root_entries(tracked_root, allowlist)
    if unexpected_root:
        return (
            "Unexpected repository root entries (add to "
            "scripts/config/root_allowlist.json with review):\n- "
            + "\n- ".join(unexpected_root)
        )
    stale_root = _stale_root_entries(tracked_root, allowlist)
    if stale_root:
        return (
            "Stale repository root allowlist entries (remove from "
            "scripts/config/root_allowlist.json):\n- " + "\n- ".join(stale_root)
        )
    return None


def _git_changed_files() -> list[str]:
    base_ref = "origin/main"
    cp = subprocess.run(
        ["git", "diff", "--name-only", f"{base_ref}...HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if cp.returncode != 0:
        return []
    return [line.strip() for line in cp.stdout.splitlines() if line.strip()]


def _fail(msg: str) -> int:
    sys.stderr.write(msg + "\n")
    return 1


def _duplicate_process_directories() -> list[str]:
    duplicates: list[str] = []
    for name in CANONICAL_PROCESS_DIRECTORY_NAMES:
        canonical_path = ROOT / "docs" / name
        root_path = ROOT / name
        if root_path.is_dir():
            duplicates.append(
                f"{name}/ duplicates canonical {canonical_path.relative_to(ROOT)}"
            )
        for parent in ROOT.iterdir():
            if not parent.is_dir() or parent == ROOT / "docs":
                continue
            process_dir = parent / name
            if process_dir.is_dir():
                duplicates.append(
                    f"{process_dir.relative_to(ROOT)}/ duplicates canonical "
                    f"{canonical_path.relative_to(ROOT)}"
                )
    return duplicates


def _duplicate_source_of_truth_headings() -> list[str]:
    duplicates: list[str] = []
    for relative_path, guarded_headings in SOURCE_OF_TRUTH_HEADINGS.items():
        document = ROOT / relative_path
        if not document.exists():
            continue
        headings = [
            line[3:].strip()
            for line in document.read_text(encoding="utf-8").splitlines()
            if line.startswith("## ") and line[3:].strip() in guarded_headings
        ]
        for heading, count in Counter(headings).items():
            if count > 1:
                duplicates.append(
                    f"{relative_path}: duplicate `## {heading}` heading appears {count} times"
                )
    return duplicates


def _duplicate_adr_numbers() -> list[str]:
    adr_dir = ROOT / "docs" / "adr"
    if not adr_dir.exists():
        return []

    duplicate_map: dict[str, list[str]] = {}
    for path in adr_dir.glob("*.md"):
        match = re.match(r"^(?P<number>\d{4})-", path.name)
        if not match:
            continue
        duplicate_map.setdefault(match.group("number"), []).append(path.name)

    return [
        f"duplicate ADR number {number}: {', '.join(sorted(file_names))}"
        for number, file_names in sorted(duplicate_map.items())
        if len(file_names) > 1
    ]


def _dangling_examples_entries() -> list[str]:
    index_path = ROOT / "docs" / "examples" / "index.rst"
    if not index_path.exists():
        return []

    missing: list[str] = []
    in_toctree = False
    for raw_line in index_path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if stripped == ".. toctree::":
            in_toctree = True
            continue
        if not in_toctree:
            continue
        if not raw_line.startswith((" ", "\t")):
            if stripped:
                in_toctree = False
            continue
        if not stripped or stripped.startswith(":"):
            continue
        entry = stripped.split()[0]
        entry_path = (ROOT / "docs" / "examples" / entry).resolve()
        candidates = (
            entry_path.with_suffix(".rst"),
            entry_path.with_suffix(".md"),
            entry_path / "index.rst",
            entry_path / "index.md",
        )
        if not any(candidate.exists() for candidate in candidates):
            missing.append(entry)
    return missing


def main() -> int:
    missing = [str(p.relative_to(ROOT)) for p in REQUIRED_FILES if not p.exists()]
    if missing:
        return _fail(
            "Missing required docs governance files:\n- " + "\n- ".join(missing)
        )
    duplicates = _duplicate_process_directories()
    if duplicates:
        return _fail(
            "Duplicate root process directories detected:\n- " + "\n- ".join(duplicates)
        )
    root_failure = _root_allowlist_failure()
    if root_failure:
        return _fail(root_failure)
    duplicate_headings = _duplicate_source_of_truth_headings()
    if duplicate_headings:
        return _fail(
            "Duplicate source-of-truth documentation headings detected:\n- "
            + "\n- ".join(duplicate_headings)
        )
    duplicate_adr_numbers = _duplicate_adr_numbers()
    if duplicate_adr_numbers:
        return _fail(
            "Duplicate ADR numbering detected:\n- " + "\n- ".join(duplicate_adr_numbers)
        )
    dangling_examples = _dangling_examples_entries()
    if dangling_examples:
        return _fail(
            "docs/examples/index.rst references missing example pages:\n- "
            + "\n- ".join(dangling_examples)
        )

    changed = _git_changed_files()
    changed_set = set(changed)

    assessment_changes = [
        p
        for p in changed
        if p.startswith("docs/assessments/") and p != "docs/assessments/README.md"
    ]
    if assessment_changes and "docs/assessments/README.md" not in changed_set:
        return _fail(
            "docs/assessments changes detected without updating docs/assessments/README.md"
        )

    adr_changes = [
        p
        for p in changed
        if p.startswith("docs/adr/")
        and p not in {"docs/adr/README.md", "docs/adr/ADR_TEMPLATE.md"}
    ]
    if adr_changes and "docs/adr/README.md" not in changed_set:
        return _fail("ADR changes detected without updating docs/adr/README.md")

    sys.stdout.write("docs governance checks passed\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
