"""Tests for scripts/ci/check_deprecated_alias_shims.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.ci import check_deprecated_alias_shims as mod


def _write_alias_file(repo_root: Path, relative_path: str) -> None:
    alias_prefix = f"{mod.DEFAULT_ALIAS_ROOT.as_posix()}/"
    normalized_path = relative_path.removeprefix(alias_prefix)
    path = repo_root / mod.DEFAULT_ALIAS_ROOT / normalized_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# shim\n", encoding="utf-8")


def test_find_unexpected_alias_files_allows_current_shim_surface(
    tmp_path: Path,
) -> None:
    for relative_path in sorted(mod.ALLOWED_ALIAS_FILES):
        _write_alias_file(tmp_path, relative_path)

    assert mod.find_unexpected_alias_files(tmp_path) == []


def test_find_unexpected_alias_files_ignores_pycache(tmp_path: Path) -> None:
    pycache = tmp_path / mod.DEFAULT_ALIAS_ROOT / "__pycache__"
    pycache.mkdir(parents=True)
    (pycache / "__init__.cpython-311.pyc").write_bytes(b"ignored")

    assert mod.find_unexpected_alias_files(tmp_path) == []


def test_find_unexpected_alias_files_flags_new_alias_module(tmp_path: Path) -> None:
    _write_alias_file(tmp_path, "__init__.py")
    _write_alias_file(tmp_path, "new_module.py")

    assert mod.find_unexpected_alias_files(tmp_path) == [
        "src/shared/python/upstream_drift_tools/new_module.py"
    ]


def test_main_fails_when_alias_surface_grows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_alias_file(tmp_path, "__init__.py")
    _write_alias_file(tmp_path, "extra.py")

    monkeypatch.setattr(mod, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr("sys.argv", ["check"])

    assert mod.main() == 1


def test_main_passes_when_alias_package_is_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mod, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr("sys.argv", ["check"])

    assert mod.main() == 0
