"""Oracle 7: Canonical Lifecycle Vocabulary Gate (Part of #14934)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

CANONICAL_LIFECYCLE_WORDS: frozenset[str] = frozenset(
    {"QUEUED", "RUNNING", "CANCEL_REQUESTED", "SUCCEEDED", "FAILED", "CANCELLED"}
)

LEGACY_STATUS_ENUM_ALLOWLIST: frozenset[str] = frozenset(
    {
        "shared/python/motion_matching/jobs/contracts.py:JobStatus",
        "shared/python/workspace/capture_inspection_handoff.py:JobStatus",
        "shared/python/workspace/estimation_workspace.py:EstimationJobStatus",
        "tools/motion_matching/process_lifecycle.py:RunStatus",
    }
)


def _extract_status_enums_from_ast(
    file_path: Path, content: str
) -> list[tuple[str, set[str], bool]]:
    tree = ast.parse(content, filename=str(file_path))
    found: list[tuple[str, set[str], bool]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or not (
            node.name.endswith("JobStatus") or node.name.endswith("RunStatus")
        ):
            continue
        is_enum = any("Enum" in ast.unparse(b) for b in node.bases)
        members: set[str] = set()
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for t in stmt.targets:
                    if isinstance(t, ast.Name) and not t.id.startswith("_"):
                        members.add(t.id)
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                if not stmt.target.id.startswith("_"):
                    members.add(stmt.target.id)
        found.append((node.name, members, is_enum))
    return found


def _scan_all_src_status_enums(
    src_dir: Path,
) -> dict[str, tuple[Path, set[str], bool]]:
    discovered: dict[str, tuple[Path, set[str], bool]] = {}
    for py_file in sorted(src_dir.rglob("*.py")):
        content = py_file.read_text(encoding="utf-8")
        if "JobStatus" not in content and "RunStatus" not in content:
            continue
        for name, members, is_enum in _extract_status_enums_from_ast(py_file, content):
            rel = py_file.relative_to(src_dir).as_posix()
            discovered[f"{rel}:{name}"] = (py_file, members, is_enum)
    return discovered


def test_oracle7_status_enums_use_canonical_lifecycle_words() -> None:
    src_dir = Path(__file__).resolve().parents[3] / "src"
    discovered = _scan_all_src_status_enums(src_dir)
    for qname, (_, members, is_enum) in discovered.items():
        if not is_enum or qname in LEGACY_STATUS_ENUM_ALLOWLIST:
            continue
        non_canonical = members - CANONICAL_LIFECYCLE_WORDS
        assert not non_canonical, (
            f"Enum {qname} has non-canonical members: {sorted(non_canonical)}"
        )


def test_oracle7_allowlist_is_tight() -> None:
    src_dir = Path(__file__).resolve().parents[3] / "src"
    discovered = _scan_all_src_status_enums(src_dir)
    for qname in sorted(LEGACY_STATUS_ENUM_ALLOWLIST):
        assert qname in discovered, f"Allowlist entry {qname} missing in src/"
        _, members, is_enum = discovered[qname]
        assert is_enum and (members - CANONICAL_LIFECYCLE_WORDS), (
            f"Allowlist entry {qname} is now canonical; remove from allowlist!"
        )


def test_oracle7_scanner_detects_legacy_violations() -> None:
    bad_code = "from enum import Enum\nclass TaskJobStatus(str, Enum):\n    QUEUED = 'QUEUED'\n    DONE = 'DONE'\n"
    enums = _extract_status_enums_from_ast(Path("mock.py"), bad_code)
    assert len(enums) == 1 and enums[0][1] - CANONICAL_LIFECYCLE_WORDS == {"DONE"}
