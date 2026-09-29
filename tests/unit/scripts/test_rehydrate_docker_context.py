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


def _commit_dockerfile(path: Path, body: str) -> Path:
    """Initialise a repo with a single committed Dockerfile and return its path."""
    _init_git_repo(path)
    dockerfile = path / "Dockerfile"
    dockerfile.write_text(body, encoding="utf-8")
    subprocess.run(
        ["git", "add", "Dockerfile"], cwd=path, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "commit", "-m", "add dockerfile"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    return dockerfile


def test_rehydrate_restores_truncated_tracked_dockerfile(tmp_path: Path) -> None:
    """A tracked Dockerfile that exists but was truncated is restored, not trusted."""
    body = "FROM python:3.12-slim\nRUN echo build\n"
    dockerfile = _commit_dockerfile(tmp_path, body)

    # A shared/partially-restored workspace can leave the file present but empty.
    dockerfile.write_text("", encoding="utf-8")

    failures = rehydrator.rehydrate_tracked_files(tmp_path, ["Dockerfile"])
    assert failures == []
    assert dockerfile.read_text(encoding="utf-8") == body


def test_rehydrate_restores_stale_tracked_dockerfile(tmp_path: Path) -> None:
    """A tracked Dockerfile whose contents drifted from HEAD is restored."""
    body = "FROM python:3.12-slim\nRUN echo build\n"
    dockerfile = _commit_dockerfile(tmp_path, body)

    dockerfile.write_text("FROM scratch\n", encoding="utf-8")

    failures = rehydrator.rehydrate_tracked_files(tmp_path, ["Dockerfile"])
    assert failures == []
    assert dockerfile.read_text(encoding="utf-8") == body


def test_check_only_reports_stale_tracked_dockerfile(tmp_path: Path) -> None:
    """``check_only`` reports content drift instead of silently passing."""
    dockerfile = _commit_dockerfile(tmp_path, "FROM python:3.12-slim\n")
    dockerfile.write_text("FROM scratch\n", encoding="utf-8")

    failures = rehydrator.rehydrate_tracked_files(
        tmp_path, ["Dockerfile"], check_only=True
    )
    assert failures == ["Tracked file 'Dockerfile' differs from HEAD"]
    # check_only must not mutate the working tree.
    assert dockerfile.read_text(encoding="utf-8") == "FROM scratch\n"


def test_rehydrate_reports_empty_tracked_blob(tmp_path: Path) -> None:
    """A tracked-but-empty Dockerfile is a failure, not just a log warning."""
    _commit_dockerfile(tmp_path, "")

    failures = rehydrator.rehydrate_tracked_files(tmp_path, ["Dockerfile"])
    assert failures == ["Tracked file 'Dockerfile' is 0 bytes at HEAD"]


def test_unchanged_tracked_dockerfile_passes_without_restore(tmp_path: Path) -> None:
    """A pristine tracked Dockerfile is left exactly as-is."""
    body = "FROM python:3.12-slim\n"
    dockerfile = _commit_dockerfile(tmp_path, body)

    failures = rehydrator.rehydrate_tracked_files(tmp_path, ["Dockerfile"])
    assert failures == []
    assert dockerfile.read_text(encoding="utf-8") == body
