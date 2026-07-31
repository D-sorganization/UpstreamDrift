"""Guard tests for the monolith refactor register (issue #7131).

The register (docs/development/monolith_refactor_register.md) is the tracked
surface for the A-O monolith findings. These tests keep it honest: it must be
in sync with the current tree, so a new oversized source file cannot land
unrecorded and a shrunk/removed file cannot linger.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "gen_monolith_register.py"
_REGISTER = _REPO_ROOT / "docs" / "development" / "monolith_refactor_register.md"


def _load():
    spec = importlib.util.spec_from_file_location("_gen_monolith_register", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = _load()


def test_register_exists() -> None:
    assert (
        _REGISTER.exists()
    ), "docs/development/monolith_refactor_register.md must exist (issue #7131)"


def test_register_is_in_sync() -> None:
    """The committed register must match the current oversized-file set."""
    assert gen.check() == 0, (
        "Monolith register is out of sync; run "
        "`python3 scripts/gen_monolith_register.py --write`"
    )


def test_threshold_is_documented() -> None:
    text = _REGISTER.read_text(encoding="utf-8")
    assert f"{gen.THRESHOLD} LOC" in text


def test_oversized_files_excludes_tests_and_vendor() -> None:
    for _loc, rel in gen.oversized_files():
        normalized = "/" + rel
        assert "/tests/" not in normalized, rel
        assert "/vendor/" not in normalized, rel
        assert rel.startswith("src/"), rel


def test_oversized_rows_are_sorted_descending() -> None:
    rows = gen.oversized_files()
    locs = [loc for loc, _ in rows]
    assert locs == sorted(locs, reverse=True)
    assert all(loc > gen.THRESHOLD for loc in locs)
