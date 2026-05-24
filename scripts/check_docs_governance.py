#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from collections import Counter
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
