"""The committed 7-iron document is current with its club marker file (#12210)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.set_spec_club_markers import read_receipt_bytes, set_document_club_markers

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs" / "development" / "full_body_models"
IRON_MARKERS = DOCS / "evidence" / "club_markers" / "iron7.json"


def test_committed_iron_document_is_current(tmp_path: Path) -> None:
    copy = tmp_path / "iron7.json"
    shutil.copy(DOCS / "full_body_spec_anthro_iron7.json", copy)
    before = copy.read_bytes()
    assert set_document_club_markers(IRON_MARKERS, copy) is False
    assert copy.read_bytes() == before


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip()


def test_receipt_commit_reads_the_pinned_receipt(tmp_path: Path) -> None:
    """``--receipt-commit`` must derive from that revision, not the worktree."""
    repo = tmp_path / "repo"
    receipt = repo / "evidence" / "receipt.json"
    receipt.parent.mkdir(parents=True)
    _git(repo.parent, "init", "-q", str(repo))
    receipt.write_bytes(b'{"pinned": true}\n')
    _git(repo, "add", ".")
    _git(
        repo,
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@t",
        "commit",
        "-q",
        "-m",
        "pin",
    )
    commit = _git(repo, "rev-parse", "HEAD")
    receipt.write_bytes(b'{"regenerated": true}\n')

    pinned = read_receipt_bytes(receipt, commit, repo_root=repo)
    assert pinned == b'{"pinned": true}\n'
    assert read_receipt_bytes(receipt, None, repo_root=repo) == (
        b'{"regenerated": true}\n'
    )
    with pytest.raises(ValueError, match="receipt-commit"):
        read_receipt_bytes(receipt, "0" * 40, repo_root=repo)


def test_refuses_a_document_for_another_club(tmp_path: Path) -> None:
    copy = tmp_path / "driver.json"
    shutil.copy(DOCS / "full_body_spec_anthro_driver.json", copy)
    with pytest.raises(ValueError, match="driver document, not iron7"):
        set_document_club_markers(IRON_MARKERS, copy)
