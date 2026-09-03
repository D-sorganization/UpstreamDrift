"""``scripts/ci/import_smoke.py`` must pass in-process (#9409, RM #1507).

The always-on ``unit-core-always`` lane runs the script as a subprocess;
this test keeps it green in the ordinary unit suite too, so a broken alias
root or a stale ``ABSENT_ALIAS_ROOTS`` entry is caught before the lane is.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "ci" / "import_smoke.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_import_smoke_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_import_smoke_main_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_script()
    exit_code = module.main()
    out = capsys.readouterr().out
    assert exit_code == 0, out
    assert "0 unexpected absent; 0 failed" in out, out


def test_absent_allowlist_entries_are_documented_shared_roots() -> None:
    """Every allowlisted root must be a real alias root with a stated reason."""
    module = _load_script()
    from src.shared.python.import_aliases import _SHARED_ROOTS

    for root, reason in module.ABSENT_ALIAS_ROOTS.items():
        assert root in _SHARED_ROOTS, root
        assert reason.strip(), f"allowlist entry {root!r} needs a reason"
