"""The collate step survives one refused fragment (RM#2093, RM#2044).

Runs the workflow's own ``Collate fragments`` script with stubbed ``gh`` and
``python3``: a refused middle fragment must stay in place while its neighbours
are collated and removed, and the job must fail only after everything else
was processed (the final gate runs after the PR-proposal step).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

WORKFLOW = (
    Path(__file__).resolve().parents[3]
    / ".github"
    / "workflows"
    / "collate-changes.yml"
)
GATE = "Fail when a fragment was refused"


def _find_bash() -> str | None:
    """Git for Windows' bash; the PATH one may be the WSL launcher."""
    git = shutil.which("git")
    if os.name == "nt" and git:
        candidate = Path(git).resolve().parent.parent / "bin" / "bash.exe"
        if candidate.is_file():
            return str(candidate)
    return shutil.which("bash")


BASH = _find_bash()
pytestmark = pytest.mark.skipif(BASH is None, reason="bash required")


def _steps() -> list[dict[str, Any]]:
    data = yaml.safe_load(WORKFLOW.read_text("utf-8"))
    return cast("list[dict[str, Any]]", data["jobs"]["collate"]["steps"])


def _step(name: str) -> dict[str, Any]:
    return next(s for s in _steps() if s.get("name") == name)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(repo), *args],
        check=True,
        capture_output=True,
    )


def _fake_bin(tmp_path: Path) -> Path:
    """``gh`` names PR 7 for every commit; ``python3`` refuses ``*-bad.md``."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "gh").write_bytes(b"#!/bin/sh\necho 7\n")
    (bin_dir / "python3").write_bytes(
        b'#!/bin/sh\nfor a in "$@"; do last="$a"; done\n'
        b'case "$last" in *-bad.md) exit 1 ;; esac\nrm -f "$last"\n'
    )
    return bin_dir


def _run_collate(tmp_path: Path, names: list[str]) -> tuple[int, str, Path, str]:
    repo = tmp_path / "repo"
    (repo / "changes").mkdir(parents=True)
    (repo / "shared_scripts").mkdir()
    (repo / "shared_scripts" / "changes_fragment.py").write_text("")
    _git(repo.parent, "init", "-q", str(repo))
    _git(repo, "commit", "-q", "--allow-empty", "-m", "base")
    for name in names:
        (repo / "changes" / name).write_text("x\n")
        _git(repo, "add", ".")
        _git(repo, "commit", "-q", "-m", f"add {name} (#7)")
    env_file = tmp_path / "env"
    env_file.write_text("")
    env = dict(os.environ)
    env["PATH"] = str(_fake_bin(tmp_path)) + os.pathsep + env["PATH"]
    env.update(GITHUB_ENV=str(env_file), REPO_SLUG="o/r")
    result = subprocess.run(
        [cast("str", BASH), "-c", str(_step("Collate fragments")["run"])],
        cwd=repo,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    return (
        result.returncode,
        result.stdout + result.stderr,
        repo,
        env_file.read_text(),
    )


def test_a_refused_fragment_is_reported_and_the_rest_still_collate(
    tmp_path: Path,
) -> None:
    code, output, repo, env = _run_collate(
        tmp_path, ["1-good.md", "2-bad.md", "3-good.md"]
    )
    assert code == 0, output
    assert "COLLATED=2" in env and "FAILED=1" in env
    assert "::error file=changes/2-bad.md::" in output
    assert sorted(p.name for p in (repo / "changes").iterdir()) == ["2-bad.md"]


def test_a_clean_run_reports_no_failures(tmp_path: Path) -> None:
    code, output, _repo, env = _run_collate(tmp_path, ["1-good.md"])
    assert code == 0, output
    assert "FAILED=0" in env


@pytest.mark.parametrize(("failed", "code"), [("0", 0), ("2", 1)])
def test_the_job_fails_after_proposing_when_a_fragment_was_refused(
    failed: str, code: int
) -> None:
    names = [str(s.get("name")) for s in _steps()]
    assert names.index("Propose collation as a pull request") < names.index(GATE)
    env = dict(os.environ, FAILED=failed)
    result = subprocess.run(
        [cast("str", BASH), "-c", str(_step(GATE)["run"])],
        capture_output=True,
        env=env,
        check=False,
    )
    assert result.returncode == code
