"""SPEC.md freshness accepts a valid change fragment in place of a SPEC edit.

Repository_Management#1976: a pull request that changes source files passes
the freshness gate when it edits SPEC.md *or* carries at least one change
fragment and every fragment validates -- the same rule as
``_has_valid_fragment`` in Repository_Management's ``fleet_hooks.py``. Any
invalid fragment, or a checkout without the vendored fragment module, falls
back to requiring the SPEC.md edit.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "ci" / "check_spec_freshness.py"
VALID = '---\nissue: 1976\nsummary: "Vendor change fragments"\n---\n'
INVALID = "---\nissue: 1976\n---\n"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_spec_freshness", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


freshness = _load()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    shutil.copytree(REPO_ROOT / "shared_scripts", tmp_path / "shared_scripts")
    (tmp_path / "changes").mkdir()
    return tmp_path


def _write(root: Path, relative: str, text: str) -> str:
    (root / relative).write_text(text, encoding="utf-8")
    return relative


def test_source_change_without_spec_or_fragment_needs_update(repo: Path) -> None:
    outputs = freshness.decide(["src/a.py"], repo)
    assert outputs["source_changed"] is True
    assert outputs["needs_update"] is True


def test_source_change_with_spec_edit_passes(repo: Path) -> None:
    assert freshness.decide(["src/a.py", "SPEC.md"], repo)["needs_update"] is False


def test_fragment_only_change_passes(repo: Path) -> None:
    fragment = _write(repo, "changes/1976-vendor.md", VALID)
    outputs = freshness.decide(["src/a.py", "tests/test_a.py", fragment], repo)
    assert outputs["fragment_valid"] is True
    assert outputs["spec_changed"] is False
    assert outputs["needs_update"] is False


def test_invalid_fragment_does_not_count(repo: Path) -> None:
    fragment = _write(repo, "changes/1976-vendor.md", INVALID)
    assert freshness.decide(["src/a.py", fragment], repo)["needs_update"] is True


def test_one_invalid_fragment_spoils_a_valid_one(repo: Path) -> None:
    good = _write(repo, "changes/1976-good.md", VALID)
    bad = _write(repo, "changes/1976-bad.md", INVALID)
    assert freshness.decide(["src/a.py", good, bad], repo)["needs_update"] is True


def test_readme_and_deleted_fragments_are_not_fragments(repo: Path) -> None:
    readme = _write(repo, "changes/README.md", "# Change Fragments\n")
    files = ["src/a.py", readme, "changes/1976-deleted.md"]
    assert freshness.decide(files, repo)["needs_update"] is True


def test_missing_fragment_module_fails_closed(repo: Path) -> None:
    fragment = _write(repo, "changes/1976-vendor.md", VALID)
    (repo / "shared_scripts" / "changes_fragment.py").unlink()
    assert freshness.decide(["src/a.py", fragment], repo)["needs_update"] is True


def test_non_source_change_needs_nothing(repo: Path) -> None:
    outputs = freshness.decide(["docs/guide.md", "scripts/x.py"], repo)
    assert outputs["source_changed"] is False
    assert outputs["needs_update"] is False


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def test_cli_reports_a_fragment_only_pull_request_as_fresh(repo: Path) -> None:
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "src").mkdir()
    _write(repo, "src/a.py", "VALUE = 1\n")
    _write(repo, "changes/1976-vendor.md", VALID)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "change")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--repo-root", str(repo), "--base", base],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "source_changed=true",
        "spec_changed=false",
        "fragment_valid=true",
        "needs_update=false",
    ]
