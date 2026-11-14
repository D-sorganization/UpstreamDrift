#!/usr/bin/env python3
"""Fail if SPEC.md or curated documentation cites repo paths that don't exist."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "SPEC.md"
HISTORICAL_PATHS = ROOT / "scripts/config/spec_historical_paths.json"

# Match `path/like/this.py` or `path/like/this/` inside backticks.
PATH_PATTERN = re.compile(r"`([a-zA-Z_][a-zA-Z0-9_./-]+(?:\.py|/))`")

FENCE_PATTERN = re.compile(r"^```.*?^```", re.MULTILINE | re.DOTALL)
INLINE_CODE_PATTERN = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
DOC_REPO_PATH_PATTERN = re.compile(r"^[a-zA-Z0-9_./-]+$")
MODULE_ATTR_SUFFIX_PATTERN = re.compile(r"\.[A-Z_][A-Z0-9_]*$")

# These prefixes are the project's package roots. Anything matched against
# them must exist; everything else is treated as a non-path identifier.
REPO_ROOTS = {
    "src/",
    "tests/",
    "scripts/",
    "ui/",
    "rust_core/",
    "shared/",
    "vendor/",
    "docs/",
    "installer/",
    ".github/",
}


def _git_bytes(root: Path, *args: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], stderr=subprocess.DEVNULL
    )


def _historical_path_is_verified(
    root: Path,
    path: str,
    row: str,
    in_change_log: bool,
    records: dict[str, dict[str, str]],
) -> bool:
    """Resolve one immutable historical row without exempting current claims."""
    record = records.get(path)
    if not in_change_log or not record:
        return False
    try:
        if hashlib.sha256(row.encode()).hexdigest() != record["row_sha256"]:
            return False
        historical = record["historical_commit"]
        pin = record["provider_commit"]
        if not all(re.fullmatch(r"[0-9a-f]{40}", value) for value in (historical, pin)):
            return False
        provider = root / "vendor/ud-tools"
        target = provider / record["provider_path"]
        if not target.resolve().is_relative_to(provider.resolve()):
            return False
        if (
            _git_bytes(root, "rev-parse", "HEAD:vendor/ud-tools").decode().strip()
            != pin
        ):
            return False
        if _git_bytes(provider, "rev-parse", "HEAD").decode().strip() != pin:
            return False
        original = _git_bytes(root, "show", f"{historical}:{path}")
        published = _git_bytes(provider, "show", f"{pin}:{record['provider_path']}")
        return (
            hashlib.sha256(original).hexdigest() == record["historical_sha256"]
            and hashlib.sha256(published).hexdigest() == record["provider_sha256"]
            and target.read_bytes() == published
        )
    except (KeyError, OSError, ValueError, subprocess.CalledProcessError):
        return False


def _default_doc_targets(root: Path) -> list[Path]:
    """Return curated architecture, testing, and user guide markdown paths."""
    targets: list[Path] = [
        root / "docs/README.md",
        root / "docs/architecture/C4.md",
        root / "docs/user_guide/mesh_backends_setup.md",
    ]
    for rel_dir in (
        "docs/testing",
        "docs/user_guide/body_part_viz",
        "docs/user_guide/upstream_drift_user_manual",
    ):
        folder = root / rel_dir
        if folder.is_dir():
            targets.extend(sorted(folder.glob("*.md")))
    return targets


def check_doc_paths(root: Path, doc_files: list[Path] | None = None) -> list[str]:
    """Return formatted missing-path violations in curated markdown files.

    Preconditions:
        ``root`` is the repository root path to resolve citations against.
    Postconditions:
        Returns an empty list iff every inline backticked repo path outside
        fenced code blocks resolves on disk under ``root``.
    """
    targets = _default_doc_targets(root) if doc_files is None else doc_files
    bad: list[str] = []
    for doc_file in targets:
        if not doc_file.is_file():
            continue
        raw = doc_file.read_text(encoding="utf-8")
        stripped = FENCE_PATTERN.sub(
            lambda match: "\n" * match.group(0).count("\n"), raw
        )
        rel = doc_file.relative_to(root).as_posix()
        for match in INLINE_CODE_PATTERN.finditer(stripped):
            path = match.group(1)
            if not any(path.startswith(prefix) for prefix in REPO_ROOTS):
                continue
            if not DOC_REPO_PATH_PATTERN.fullmatch(path):
                continue
            if MODULE_ATTR_SUFFIX_PATTERN.search(path):
                continue
            if not (root / path.rstrip("/")).exists():
                line = stripped[: match.start()].count("\n") + 1
                bad.append(f"  {rel}:{line}: missing path {path!r}")
    return bad


def main() -> int:
    text = SPEC.read_text(encoding="utf-8")
    records = json.loads(HISTORICAL_PATHS.read_text(encoding="utf-8"))["paths"]
    spec_bad: list[str] = []
    history_header = re.search(r"(?m)^## 12\. Change Log\s*$", text)
    history_start = history_header.end() if history_header else len(text) + 1
    history_end = text.find("\n## ", history_start)
    if history_end < 0:
        history_end = len(text)
    for match in PATH_PATTERN.finditer(text):
        path = match.group(1)
        if not any(path.startswith(prefix) for prefix in REPO_ROOTS):
            continue
        target = ROOT / path.rstrip("/")
        if not target.exists():
            line = text[: match.start()].count("\n") + 1
            row = text.splitlines()[line - 1]
            if _historical_path_is_verified(
                ROOT, path, row, history_start <= match.start() < history_end, records
            ):
                print(f"Verified historical path provenance: SPEC.md:{line}: {path}")
                continue
            spec_bad.append(f"  SPEC.md:{line}: missing path {path!r}")
    doc_bad = check_doc_paths(ROOT)
    if spec_bad:
        print("FAIL: SPEC.md references paths that do not exist:")
        print("\n".join(spec_bad))
        print("\nUpdate SPEC.md or add the file/directory it claims exists.")
    if doc_bad:
        print("FAIL: Documentation files reference paths that do not exist:")
        print("\n".join(doc_bad))
        print(
            "\nUpdate the documentation path or add the file/directory it claims exists."
        )
    if spec_bad or doc_bad:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
