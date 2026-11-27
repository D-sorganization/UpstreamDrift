"""Tests for the motion-matching leaderboard freshness gate (issue #14767).

A squash merge orphans the ``git_head`` that ``run_leaderboard.m`` wrote on
the PR branch. These tests rebuild that history in a throwaway repository.
"""

from __future__ import annotations

from importlib import util
import json
from pathlib import Path
import subprocess

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "ci" / "check_leaderboard_freshness.py"
MM = "matlab/motion_matching"
META = f"{MM}/results/.leaderboard_meta.json"


def _load():
    spec = util.spec_from_file_location("check_leaderboard_freshness", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write(repo: Path, rel: str, text: str) -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture()
def squashed(tmp_path: Path) -> tuple[Path, str]:
    """Main whose sidecar pins a PR-branch commit lost to a squash merge."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "ci@example.invalid")
    _git(repo, "config", "user.name", "ci")
    _write(repo, f"{MM}/option1/solve.m", "x = 1;\n")
    _commit(repo, "base")
    _git(repo, "checkout", "-q", "-b", "feature")
    _write(repo, f"{MM}/option1/solve.m", "x = 2;\n")
    regen = _commit(repo, "solver change, leaderboard run here")
    _write(repo, META, json.dumps({"git_head": regen}))
    _commit(repo, "commit sidecar")
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--squash", "feature")
    _commit(repo, "squash merge")
    _git(repo, "branch", "-q", "-D", "feature")
    _git(repo, "reflog", "expire", "--expire=now", "--all")
    _git(repo, "gc", "-q", "--prune=now")
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "-b", "pr")
    return repo, base


def _check(repo: Path, base: str | None) -> tuple[int, list[str]]:
    lines: list[str] = []
    code = _load().check(repo, MM, base, lines.append)
    return code, lines


def test_docs_only_pr_passes_despite_orphaned_git_head(
    squashed: tuple[Path, str],
) -> None:
    """A PR that changes no solver input does not need a regenerated board."""
    repo, base = squashed
    _write(repo, f"{MM}/shared/NOTES.md", "docs only\n")
    _write(repo, f"{MM}/option1/tests/test_solve.m", "assert(true);\n")
    _commit(repo, "docs and tests")
    code, lines = _check(repo, base)
    assert code == 0, lines
    assert any("no solver inputs" in line for line in lines)


def test_solver_change_still_requires_regeneration(
    squashed: tuple[Path, str],
) -> None:
    """A solver change on top of an orphaned git_head still fails loudly."""
    repo, base = squashed
    _write(repo, f"{MM}/option1/solve.m", "x = 3;\n")
    _commit(repo, "solver change without a rerun")
    code, lines = _check(repo, base)
    assert code == 1
    assert any("is not a known commit" in line for line in lines)


def test_fresh_regeneration_on_the_branch_passes(
    squashed: tuple[Path, str],
) -> None:
    """Solver change plus a sidecar pinned on this branch is fresh."""
    repo, base = squashed
    _write(repo, f"{MM}/option1/solve.m", "x = 4;\n")
    regen = _commit(repo, "solver change, rerun")
    _write(repo, META, json.dumps({"git_head": regen}))
    _commit(repo, "commit sidecar")
    assert _check(repo, base)[0] == 0


def test_solver_edit_after_regeneration_is_stale(
    squashed: tuple[Path, str],
) -> None:
    """Editing a solver file after the rerun makes the board stale again."""
    repo, base = squashed
    _write(repo, f"{MM}/option1/solve.m", "x = 5;\n")
    regen = _commit(repo, "rerun")
    _write(repo, META, json.dumps({"git_head": regen}))
    _write(repo, f"{MM}/option1/solve.m", "x = 6;\n")
    _commit(repo, "sidecar plus a later solver edit")
    code, lines = _check(repo, base)
    assert code == 1
    assert any("option1/solve.m" in line for line in lines)


def test_without_a_pr_base_the_full_contract_applies(
    squashed: tuple[Path, str],
) -> None:
    """``workflow_dispatch`` has no PR base, so ancestry is checked as before."""
    repo, _ = squashed
    code, lines = _check(repo, None)
    assert code == 1
    assert any("is not a known commit" in line for line in lines)
