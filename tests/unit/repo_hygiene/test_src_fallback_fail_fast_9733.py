"""Fail-fast guard for the uninitialized vendored Tools tree (UpstreamDrift#9733).
When ``vendor/ud-tools`` is not initialized, the ``src`` fallback machinery
previously installed its finder anyway — ``shared.python`` still resolves to
UpstreamDrift's own aliased copy under the test bootstrap — and the resulting
meta-path consultation recursed (``_installed_tools_spec`` → ``find_spec`` →
finder) without any diagnostic, livelocking pytest collection.

These tests pin the fail-fast contract at the registration boundary: an
uninitialized vendored tree with no genuine Tools distribution must raise an
actionable ImportError naming the remediation command, and no fallback finder
may be installed in that state. Every probe is simulated via monkeypatching,
so the tests never touch the real submodule state and never hang.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import src as ud_src

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_UD_SHARED = _REPO_ROOT / "src" / "shared" / "python"
_VENDORED_SHARED = _REPO_ROOT / "vendor" / "ud-tools" / "src" / "shared" / "python"
# A path the vendored-tree probe reports as absent, without touching the real
# submodule directory.
_MISSING_VENDORED_SRC = _REPO_ROOT / "vendor" / "ud-tools" / "simulated-uninitialized"
_REMEDIATION = "git submodule update --init vendor/ud-tools"


def _fallback_finder_count() -> int:
    """Count ``_VendoredToolsFallbackFinder`` instances on the meta path."""
    return sum(
        isinstance(finder, ud_src._VendoredToolsFallbackFinder)  # noqa: SLF001
        for finder in sys.meta_path
    )


def _alias_spec() -> SimpleNamespace:
    """A spec resolving ``shared.python`` to UpstreamDrift's own aliased copy."""
    return SimpleNamespace(
        origin=str(_UD_SHARED / "__init__.py"), submodule_search_locations=None
    )


def _installed_distribution_spec(tmp_path: Path) -> SimpleNamespace:
    """A spec resolving ``shared.python`` to an installed Tools distribution."""
    origin = tmp_path / "site-packages" / "shared" / "python" / "__init__.py"
    origin.parent.mkdir(parents=True, exist_ok=True)
    origin.write_text("", encoding="utf-8")
    return SimpleNamespace(origin=str(origin), submodule_search_locations=None)


@pytest.fixture
def _uninitialized_vendored_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the vendored-tree probe fail without touching the real submodule."""
    monkeypatch.setattr(ud_src, "_VENDORED_TOOLS_SRC", _MISSING_VENDORED_SRC)


def test_uninitialized_vendored_tree_raises_actionable_import_error(
    _uninitialized_vendored_tree, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no vendored tree and only the aliased copy, registration must raise."""
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda *args, **kwargs: _alias_spec()
    )
    with pytest.raises(ImportError, match=_REMEDIATION):
        ud_src._register_vendored_tools_fallback()


def test_the_guard_fires_before_any_fallback_finder_is_installed(
    _uninitialized_vendored_tree, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The recursion driver must never reach ``sys.meta_path`` in this state."""
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda *args, **kwargs: _alias_spec()
    )
    before = _fallback_finder_count()
    with pytest.raises(ImportError, match=_REMEDIATION):
        ud_src._register_vendored_tools_fallback()
    assert _fallback_finder_count() == before


def test_an_installed_tools_distribution_still_registers_without_the_vendored_tree(
    _uninitialized_vendored_tree, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A genuine external ``shared.python`` keeps the fallback working (#5048)."""
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda *args, **kwargs: _installed_distribution_spec(tmp_path),
    )
    assert ud_src._register_vendored_tools_fallback() is True


def test_no_tools_anywhere_stays_graceful(
    _uninitialized_vendored_tree, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkout with no Tools dependency at all keeps today's soft skip."""
    monkeypatch.setattr(importlib.util, "find_spec", lambda *args, **kwargs: None)
    assert ud_src._register_vendored_tools_fallback() is False


def test_initialized_vendored_tree_registers_the_fallback() -> None:
    """The real initialized worktree keeps the fallback registered."""
    if not _VENDORED_SHARED.is_dir():
        pytest.skip("vendor/ud-tools is not initialized in this checkout")
    assert ud_src._register_vendored_tools_fallback() is True


def test_the_guard_survives_python_optimize() -> None:
    """The fail-fast raise must be an explicit raise, not an ``assert``."""
    if not _VENDORED_SHARED.is_dir():
        pytest.skip("vendor/ud-tools is not initialized in this checkout")
    script = f"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import src as ud_src

ud_src._VENDORED_TOOLS_SRC = Path({_MISSING_VENDORED_SRC.as_posix()!r})
alias_origin = Path({(_UD_SHARED / "__init__.py").as_posix()!r})
importlib.util.find_spec = lambda *a, **k: SimpleNamespace(
    origin=str(alias_origin), submodule_search_locations=None
)
try:
    ud_src._register_vendored_tools_fallback()
except ImportError as exc:
    assert {_REMEDIATION!r} in str(exc), str(exc)
    print("GUARD_RAISED")
else:
    raise SystemExit("no ImportError raised under -O")
"""
    proc = subprocess.run(
        [sys.executable, "-O", "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=_REPO_ROOT,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "GUARD_RAISED" in proc.stdout
