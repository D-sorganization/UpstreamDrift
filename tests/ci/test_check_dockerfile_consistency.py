"""Tests for the Dockerfile pip-pin consistency check (issue #7161)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "ci"
    / "check_dockerfile_consistency.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "check_dockerfile_consistency", _SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_repo_dockerfiles_are_consistent() -> None:
    mod = _load_module()
    repo_root = Path(__file__).resolve().parents[2]
    assert mod.check(repo_root) == []


def test_detects_divergent_pins(tmp_path: Path) -> None:
    mod = _load_module()
    (tmp_path / "Dockerfile").write_text(
        "RUN pip install --upgrade pip==26.1\n", encoding="utf-8"
    )
    (tmp_path / "Dockerfile.modular").write_text(
        "RUN pip install --upgrade pip==25.3\n", encoding="utf-8"
    )
    errors = mod.check(tmp_path)
    assert errors
    assert any("Inconsistent pip pins" in e for e in errors)


def test_consistent_pins_pass(tmp_path: Path) -> None:
    mod = _load_module()
    (tmp_path / "Dockerfile").write_text(
        "RUN pip install pip==26.1\n", encoding="utf-8"
    )
    (tmp_path / "Dockerfile.modular").write_text(
        "RUN pip install pip==26.1\n", encoding="utf-8"
    )
    assert mod.check(tmp_path) == []
