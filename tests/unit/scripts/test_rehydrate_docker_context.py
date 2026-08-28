"""Tests for Dockerfile and build context rehydration validation."""

from __future__ import annotations

from pathlib import Path
import subprocess
import pytest

from scripts.ci import rehydrate_docker_context as rehydrator

pytestmark = pytest.mark.unit


def _init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=path,
        check=True,
        capture_output=True,
    )


def test_rehydrate_missing_tracked_dockerfile_restores_file(tmp_path: Path) -> None:
    """When a tracked Dockerfile is missing from the working tree, rehydration restores it from HEAD."""
    _init_git_repo(tmp_path)
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM python:3.12-slim\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "Dockerfile"], cwd=tmp_path, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "commit", "-m", "add dockerfile"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    # Simulate disappearance in dirty shared workspace
    dockerfile.unlink()
    assert not dockerfile.exists()

    failures = rehydrator.rehydrate_tracked_files(tmp_path, ["Dockerfile"])
    assert failures == []
    assert dockerfile.exists()
    assert dockerfile.read_text(encoding="utf-8") == "FROM python:3.12-slim\n"


def test_rehydrate_untracked_target_fails_closed(tmp_path: Path) -> None:
    """If target is not tracked at HEAD, rehydration fails closed and does not create fake file."""
    _init_git_repo(tmp_path)
    (tmp_path / "dummy.txt").write_text("dummy", encoding="utf-8")
    subprocess.run(
        ["git", "add", "dummy.txt"], cwd=tmp_path, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True
    )

    failures = rehydrator.rehydrate_tracked_files(tmp_path, ["Dockerfile"])
    assert any("not tracked at HEAD" in f for f in failures)
    assert not (tmp_path / "Dockerfile").exists()


def test_rehydrate_existing_tracked_file_passes(tmp_path: Path) -> None:
    """When tracked files exist and are intact, rehydration passes cleanly."""
    _init_git_repo(tmp_path)
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM python:3.12-slim\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "Dockerfile"], cwd=tmp_path, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "commit", "-m", "add dockerfile"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    failures = rehydrator.rehydrate_tracked_files(tmp_path, ["Dockerfile"])
    assert failures == []


def test_rehydrate_check_only_reports_missing_without_restoring(tmp_path: Path) -> None:
    """--check-only reports missing tracked files without writing to disk."""
    _init_git_repo(tmp_path)
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM python:3.12-slim\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "Dockerfile"], cwd=tmp_path, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "commit", "-m", "add dockerfile"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    dockerfile.unlink()
    failures = rehydrator.rehydrate_tracked_files(
        tmp_path, ["Dockerfile"], check_only=True
    )
    assert any("missing on disk" in f for f in failures)
    assert not dockerfile.exists()
