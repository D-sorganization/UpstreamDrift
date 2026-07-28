from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts import update_biomech_vendor

pytestmark = pytest.mark.unit


def _run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _make_source_repo(tmp_path: Path) -> tuple[Path, str]:
    source = tmp_path / "source"
    source.mkdir()
    _run_git(source, "init", "-b", "main")
    _run_git(source, "config", "user.email", "test@example.invalid")
    _run_git(source, "config", "user.name", "Test User")
    models = source / "models"
    models.mkdir()
    (models / "model.txt").write_text("model data\n", encoding="utf-8")
    _run_git(source, "add", "models/model.txt")
    _run_git(source, "commit", "-m", "seed model")
    return source, _run_git(source, "rev-parse", "HEAD")


def test_snapshot_records_resolved_commit_for_mutable_ref(
    tmp_path: Path,
) -> None:
    source, commit_sha = _make_source_repo(tmp_path)
    options = update_biomech_vendor.VendorOptions(
        repo="MuJoCo_Models",
        ref="main",
        url=str(source),
        vendor_root=tmp_path / "vendor",
        allow_mutable_ref=True,
    )

    destination = update_biomech_vendor.snapshot(options)

    provenance = (destination / "VENDOR_PROVENANCE.txt").read_text(encoding="utf-8")
    assert f"commit: {commit_sha}" in provenance
    assert "ref: main" in provenance
    assert "url_override: true" in provenance
    assert "mutable_ref_allowed: true" in provenance
    assert "working_tree_clean: true" in provenance


def test_snapshot_accepts_full_commit_sha_without_mutable_ref_opt_in(
    tmp_path: Path,
) -> None:
    source, commit_sha = _make_source_repo(tmp_path)
    options = update_biomech_vendor.VendorOptions(
        repo="MuJoCo_Models",
        ref=commit_sha,
        url=str(source),
        vendor_root=tmp_path / "vendor",
    )

    destination = update_biomech_vendor.snapshot(options)

    provenance = (destination / "VENDOR_PROVENANCE.txt").read_text(encoding="utf-8")
    assert f"ref: {commit_sha}" in provenance
    assert f"commit: {commit_sha}" in provenance
    assert "mutable_ref_allowed: false" in provenance


def test_mutable_ref_requires_explicit_opt_in(tmp_path: Path) -> None:
    source, _commit_sha = _make_source_repo(tmp_path)
    options = update_biomech_vendor.VendorOptions(
        repo="MuJoCo_Models",
        ref="main",
        url=str(source),
        vendor_root=tmp_path / "vendor",
        allow_mutable_ref=False,
    )

    with pytest.raises(ValueError, match="full 40-character commit SHA"):
        update_biomech_vendor.snapshot(options)
