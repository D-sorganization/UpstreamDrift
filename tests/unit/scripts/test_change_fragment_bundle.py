"""The vendored change-fragment bundle must stay byte-identical and work here.

Repository_Management#1976 vendors ``shared_scripts/changes_fragment.py`` and
the sibling modules it imports from Repository_Management so pull requests can
ship a per-PR fragment instead of editing SPEC.md and the development log.
``docs/development/change-fragment-bundle.json`` records the upstream digests;
a local edit or a formatter rewrite would silently fork the fleet copy, which
is how issue #9476 shipped a false claim into three repositories at once.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

REPO_ROOT = Path(__file__).resolve().parents[3]
RECEIPT = REPO_ROOT / "docs" / "development" / "change-fragment-bundle.json"
CLI = Path("shared_scripts") / "changes_fragment.py"


def _receipt() -> dict[str, object]:
    data = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def test_receipt_names_the_upstream_source() -> None:
    receipt = _receipt()
    assert receipt["source_repository"] == "D-sorganization/Repository_Management"
    commit = receipt["source_commit"]
    assert isinstance(commit, str) and len(commit) == 40


def test_receipt_covers_every_module_the_cli_needs() -> None:
    files = _receipt()["files"]
    assert isinstance(files, dict)
    expected = {
        f"shared_scripts/{name}.py"
        for name in (
            "changes_fragment",
            "changes_fragment_schema",
            "changes_fragment_collate",
            "development_log",
            "development_log_schema",
            "development_log_validator",
            "document_title_case",
        )
    }
    assert set(files) == expected


def test_vendored_files_are_byte_identical_to_the_receipt() -> None:
    files = _receipt()["files"]
    assert isinstance(files, dict)
    for relative, digest in files.items():
        actual = hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()
        assert actual == digest, f"{relative} drifted from Repository_Management"


@pytest.fixture
def scratch_repo(tmp_path: Path) -> Path:
    """A copy of the shared files the collator edits, plus the vendored tools."""
    shutil.copytree(REPO_ROOT / "shared_scripts", tmp_path / "shared_scripts")
    shutil.copyfile(REPO_ROOT / "SPEC.md", tmp_path / "SPEC.md")
    devlog = Path("docs") / "development" / "DEVELOPMENT_LOG.md"
    (tmp_path / devlog).parent.mkdir(parents=True)
    shutil.copyfile(REPO_ROOT / devlog, tmp_path / devlog)
    return tmp_path


def _cli(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CLI), "--repo-root", str(root), *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def test_new_validate_collate_round_trip(scratch_repo: Path) -> None:
    created = _cli(
        scratch_repo,
        "new",
        "--issue",
        "1976",
        "--summary",
        "vendor the change fragment tooling",
        "--dl-state",
        "in_review",
        "--next-step",
        "Merge the pull request.",
        "--branch",
        "chore/1976-change-fragments-tooling",
    )
    assert created.returncode == 0, created.stderr
    fragment = scratch_repo / "changes" / "1976-vendor-the-change-fragment-tooling.md"
    assert fragment.is_file()

    validated = _cli(scratch_repo, "validate")
    assert validated.returncode == 0, validated.stderr

    collated = _cli(
        scratch_repo,
        "collate",
        "--pr",
        "4242",
        "--sha",
        "abcdef1",
        "--date",
        "2026-10-04",
    )
    assert collated.returncode == 0, collated.stderr
    assert not fragment.exists()
    spec = (scratch_repo / "SPEC.md").read_text(encoding="utf-8")
    assert "| 2026-10-04 | #4242 | vendor the change fragment tooling |" in spec
    devlog = (scratch_repo / "docs/development/DEVELOPMENT_LOG.md").read_text(
        encoding="utf-8"
    )
    # Title-cased so the collated heading passes the document-title gate.
    assert "### DL-#1976 · Vendor the Change Fragment Tooling" in devlog
    assert "- **PR:** #4242" in devlog


def test_validate_rejects_a_malformed_fragment(scratch_repo: Path) -> None:
    bad = scratch_repo / "changes" / "1976-bad.md"
    bad.parent.mkdir()
    bad.write_text("---\nissue: 1976\n---\n", encoding="utf-8")
    result = _cli(scratch_repo, "validate")
    assert result.returncode == 1
    assert "summary is required" in result.stderr


def test_collate_updates_a_legacy_hyphen_entry_in_place(scratch_repo: Path) -> None:
    """``DL-#11329 - ...`` (hyphen heading) is updated, never duplicated."""
    devlog_path = scratch_repo / "docs/development/DEVELOPMENT_LOG.md"
    assert "### DL-#11329 - " in devlog_path.read_text(encoding="utf-8")
    fragment = scratch_repo / "changes" / "11329-touch.md"
    fragment.parent.mkdir()
    fragment.write_text(
        '---\nissue: 11329\nsummary: "Touch the scapula entry"\n---\n',
        encoding="utf-8",
    )
    collated = _cli(
        scratch_repo,
        "collate",
        "--pr",
        "4343",
        "--sha",
        "abcdef1",
        "--date",
        "2026-10-04",
    )
    assert collated.returncode == 0, collated.stderr
    devlog = devlog_path.read_text(encoding="utf-8")
    assert devlog.count("### DL-#11329 ") == 1
    assert "### DL-#11329 · Scapula and Quiet Torso Matching" in devlog
