"""Contracts for the JaxSim upgrade-guard script (issue #6660).

These tests exercise the pin-drift logic without importing ``jaxsim`` so they
run on every platform, including the Windows dev box.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.jaxsim.check_jaxsim_pin import (
    EXPECTED_JAXSIM_REQUIREMENT,
    check,
    read_declared_requirement,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"


def test_repo_pin_is_intact() -> None:
    """The live pyproject pin must match the guarded requirement."""
    assert read_declared_requirement(_PYPROJECT) == EXPECTED_JAXSIM_REQUIREMENT
    assert check(_PYPROJECT) == []


def _write_pyproject(tmp_path: Path, requirement: str) -> Path:
    content = (
        "[project]\n"
        'name = "x"\n'
        'version = "0"\n'
        "[project.optional-dependencies]\n"
        f'jaxsim = ["{requirement}"]\n'
    )
    path = tmp_path / "pyproject.toml"
    path.write_text(content, encoding="utf-8")
    return path


def test_drifted_pin_is_reported(tmp_path: Path) -> None:
    path = _write_pyproject(tmp_path, "jaxsim==0.10.0")
    errors = check(path)
    assert any("drifted" in e for e in errors)


def test_intact_pin_yields_no_errors(tmp_path: Path) -> None:
    path = _write_pyproject(tmp_path, EXPECTED_JAXSIM_REQUIREMENT)
    assert check(path) == []


def test_missing_extra_raises(tmp_path: Path) -> None:
    content = '[project]\nname = "x"\nversion = "0"\n'
    path = tmp_path / "pyproject.toml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(KeyError):
        read_declared_requirement(path)


def test_multiple_requirements_rejected(tmp_path: Path) -> None:
    content = (
        "[project]\n"
        'name = "x"\n'
        'version = "0"\n'
        "[project.optional-dependencies]\n"
        'jaxsim = ["jaxsim==0.9.0", "extra==1.0"]\n'
    )
    path = tmp_path / "pyproject.toml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one"):
        read_declared_requirement(path)
