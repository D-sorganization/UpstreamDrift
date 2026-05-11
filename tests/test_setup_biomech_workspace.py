"""Smoke tests for ``scripts/setup_biomech_workspace.sh``.

The bootstrap script editable-installs every sibling biomech repo that
exists at ``../<RepoName>/`` and skips the rest. We exercise the
shell-side logic by stubbing ``python3`` so no real pip install runs.

See ``docs/adr/0014-shared-biomech-models.md`` (UpstreamDrift#5184).
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "setup_biomech_workspace.sh"
)

SIBLINGS = (
    "MuJoCo_Models",
    "Drake_Models",
    "Pinocchio_Models",
    "OpenSim_Models",
    "Movement-Optimizer",
)


def _bash_available() -> bool:
    return shutil.which("bash") is not None


pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(not _bash_available(), reason="bash not on PATH"),
]


def _make_workspace(tmp_path: Path, present: list[str]) -> Path:
    """Build a workspace with a stub UpstreamDrift checkout and given siblings."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    repo_root = workspace / "UpstreamDrift"
    (repo_root / "scripts").mkdir(parents=True)
    target = repo_root / "scripts" / "setup_biomech_workspace.sh"
    shutil.copy2(SCRIPT_PATH, target)
    target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    for repo_name in present:
        sibling = workspace / repo_name
        sibling.mkdir()
        (sibling / "pyproject.toml").write_text(
            "[project]\nname = 'stub'\nversion = '0.0.1'\n",
            encoding="utf-8",
        )
    return repo_root


def _stub_python(tmp_path: Path, *, exit_code: int = 0) -> Path:
    """Return a path to a fake ``python3`` that records its invocations."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "python3"
    log_path = tmp_path / "python3_invocations.log"
    stub.write_text(
        f'#!/usr/bin/env bash\necho "$@" >> "{log_path}"\nexit {exit_code}\n',
        encoding="utf-8",
    )
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return bin_dir


def _run_script(repo_root: Path, stub_bin: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PATH"] = f"{stub_bin}{os.pathsep}{env.get('PATH', '')}"
    return subprocess.run(
        ["bash", str(repo_root / "scripts" / "setup_biomech_workspace.sh")],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX bash script smoke test")
def test_all_siblings_present(tmp_path: Path) -> None:
    """Every sibling exists → bootstrap installs all five and exits clean."""
    repo_root = _make_workspace(tmp_path, list(SIBLINGS))
    stub_bin = _stub_python(tmp_path)
    result = _run_script(repo_root, stub_bin)
    assert result.returncode == 0, result.stderr
    log = (tmp_path / "python3_invocations.log").read_text(encoding="utf-8")
    for repo_name in SIBLINGS:
        assert repo_name in log
    assert "installed: 5" in result.stdout


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX bash script smoke test")
def test_no_siblings_present(tmp_path: Path) -> None:
    """No sibling checkouts → bootstrap is a no-op with zero exit code."""
    repo_root = _make_workspace(tmp_path, [])
    stub_bin = _stub_python(tmp_path)
    result = _run_script(repo_root, stub_bin)
    assert result.returncode == 0, result.stderr
    assert "installed: 0" in result.stdout
    assert "skipped : 5" in result.stdout
    log = tmp_path / "python3_invocations.log"
    assert not log.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX bash script smoke test")
def test_partial_siblings_present(tmp_path: Path) -> None:
    """A subset of siblings → only those are installed."""
    repo_root = _make_workspace(tmp_path, ["MuJoCo_Models", "Drake_Models"])
    stub_bin = _stub_python(tmp_path)
    result = _run_script(repo_root, stub_bin)
    assert result.returncode == 0, result.stderr
    assert "installed: 2" in result.stdout
    log = (tmp_path / "python3_invocations.log").read_text(encoding="utf-8")
    assert "MuJoCo_Models" in log
    assert "Drake_Models" in log
    assert "Pinocchio_Models" not in log


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX bash script smoke test")
def test_sibling_without_pyproject_is_skipped(tmp_path: Path) -> None:
    """A sibling directory without ``pyproject.toml`` is treated as not present."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    repo_root = workspace / "UpstreamDrift"
    (repo_root / "scripts").mkdir(parents=True)
    target = repo_root / "scripts" / "setup_biomech_workspace.sh"
    shutil.copy2(SCRIPT_PATH, target)
    target.chmod(target.stat().st_mode | stat.S_IXUSR)
    (workspace / "MuJoCo_Models").mkdir()  # no pyproject.toml
    stub_bin = _stub_python(tmp_path)
    result = _run_script(repo_root, stub_bin)
    assert result.returncode == 0
    assert "installed: 0" in result.stdout
    assert "no pyproject.toml" in result.stdout


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX bash script smoke test")
def test_install_failure_propagates_exit_code(tmp_path: Path) -> None:
    """A failed pip install yields exit code 1 and is tallied as failed."""
    repo_root = _make_workspace(tmp_path, ["MuJoCo_Models"])
    stub_bin = _stub_python(tmp_path, exit_code=2)
    result = _run_script(repo_root, stub_bin)
    assert result.returncode == 1
    assert "failed   : 1" in result.stdout
