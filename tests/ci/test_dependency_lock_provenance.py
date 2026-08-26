"""Contracts for deterministic pip-tools lockfile provenance (issue #9120)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from piptools.writer import OutputWriter


REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_COMPILE_COMMAND = "make sync-deps"
LOCKFILES = ("requirements.lock", "requirements-dev.lock")


def _sync_deps_dry_run(*, offline: bool) -> bytes:
    """Return the repository recipe without executing dependency resolution."""

    environment = os.environ.copy()
    for variable in ("PIP_FIND_LINKS", "PIP_INDEX_URL", "PIP_NO_INDEX"):
        environment.pop(variable, None)
    if offline:
        environment.update(
            {
                "PIP_FIND_LINKS": "/nonexistent/offline-wheelhouse",
                "PIP_INDEX_URL": "https://invalid.example.invalid/simple",
                "PIP_NO_INDEX": "1",
            }
        )

    result = subprocess.run(
        ["make", "--no-print-directory", "--dry-run", "sync-deps"],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
    )
    return result.stdout


def _piptools_header(monkeypatch, *, offline: bool) -> bytes:
    """Construct the header using pip-tools' supported writer contract."""

    monkeypatch.setenv("CUSTOM_COMPILE_COMMAND", CANONICAL_COMPILE_COMMAND)
    if offline:
        monkeypatch.setenv("PIP_NO_INDEX", "1")
        monkeypatch.setenv("PIP_INDEX_URL", "https://invalid.example.invalid/simple")
    else:
        monkeypatch.delenv("PIP_NO_INDEX", raising=False)
        monkeypatch.delenv("PIP_INDEX_URL", raising=False)

    writer = OutputWriter.__new__(OutputWriter)
    writer.emit_header = True
    writer.click_ctx = None
    return ("\n".join(writer.write_header()) + "\n").encode("utf-8")


def test_sync_deps_recipe_is_environment_invariant() -> None:
    normal = _sync_deps_dry_run(offline=False)
    offline = _sync_deps_dry_run(offline=True)

    assert offline == normal
    assert normal.count(b'CUSTOM_COMPILE_COMMAND="make sync-deps"') == 2


def test_piptools_header_is_byte_stable_offline(monkeypatch) -> None:
    normal = _piptools_header(monkeypatch, offline=False)
    offline = _piptools_header(monkeypatch, offline=True)

    assert offline == normal
    assert b"#    make sync-deps\n" in normal


def test_committed_lockfiles_use_canonical_generated_header() -> None:
    for relative_path in LOCKFILES:
        header = (REPO_ROOT / relative_path).read_bytes().splitlines()[:8]
        assert b"#    make sync-deps" in header, relative_path
