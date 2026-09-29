"""Test for repo hygiene: every tracked gitlink must be declared in .gitmodules.

A gitlink (tree entry with mode 160000) that has no matching ``[submodule]``
stanza in ``.gitmodules`` breaks the documented fresh-clone setup step
``git submodule update --init --recursive``, which exits 128 with
"No url found for submodule path". See issue #8604.
"""

import configparser
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

ROOT = Path(__file__).resolve().parent.parent.parent.parent
GITLINK_MODE = "160000"


def _tracked_gitlink_paths() -> list[str]:
    """Return every tracked tree entry recorded as a gitlink."""
    result = subprocess.run(
        ["git", "ls-tree", "-r", "HEAD"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip("Git is not available or this is not a git repository")

    paths = []
    for line in result.stdout.splitlines():
        meta, _, path = line.partition("\t")
        if meta.split(" ", 1)[0] == GITLINK_MODE:
            paths.append(path)
    return paths


def _declared_submodule_paths() -> set[str]:
    """Return every submodule path declared in .gitmodules."""
    gitmodules = ROOT / ".gitmodules"
    if not gitmodules.exists():
        return set()

    parser = configparser.ConfigParser()
    parser.read_string(gitmodules.read_text(encoding="utf-8"))
    return {
        parser.get(section, "path")
        for section in parser.sections()
        if parser.has_option(section, "path")
    }


def test_no_orphaned_gitlinks() -> None:
    orphaned = sorted(set(_tracked_gitlink_paths()) - _declared_submodule_paths())

    if orphaned:
        listing = "\n".join(f"  - {path}" for path in orphaned)
        pytest.fail(
            "Tracked gitlinks have no .gitmodules entry:\n"
            f"{listing}\n"
            "These break `git submodule update --init --recursive` on a fresh "
            "clone. Either declare them in .gitmodules or remove them with "
            "`git rm --cached <path>`."
        )
