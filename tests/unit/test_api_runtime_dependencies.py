"""API runtime dependencies must be core, not dev-only (issue #7125)."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_LOCK = _REPO_ROOT / "requirements.lock"
_API_RUNTIME_DEPS = (
    "sqlalchemy",
    "alembic",
    "bcrypt",
    "pyjwt",
    "cryptography",
    "email-validator",
)


def _canonical(requirement: str) -> str:
    name = requirement.strip()
    for sep in ("==", ">=", "<=", "~=", "!=", ">", "<", "[", ";", " "):
        idx = name.find(sep)
        if idx != -1:
            name = name[:idx]
    return name.strip().lower().replace("_", "-")


@pytest.fixture(scope="module")
def pyproject() -> dict:
    return tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def core_dep_names(pyproject: dict) -> set[str]:
    return {_canonical(dep) for dep in pyproject["project"]["dependencies"]}


@pytest.fixture(scope="module")
def dev_dep_names(pyproject: dict) -> set[str]:
    return {
        _canonical(dep) for dep in pyproject["project"]["optional-dependencies"]["dev"]
    }


@pytest.mark.parametrize("dep", _API_RUNTIME_DEPS)
def test_api_runtime_dep_is_core(dep: str, core_dep_names: set[str]) -> None:
    assert dep in core_dep_names


@pytest.mark.parametrize("dep", _API_RUNTIME_DEPS)
def test_api_runtime_dep_not_dev_only(dep: str, dev_dep_names: set[str]) -> None:
    assert dep not in dev_dep_names


@pytest.mark.parametrize("dep", _API_RUNTIME_DEPS)
def test_api_runtime_dep_is_locked(dep: str) -> None:
    locked = {
        _canonical(line)
        for line in _LOCK.read_text(encoding="utf-8").splitlines()
        if "==" in line and not line.lstrip().startswith("#")
    }
    assert dep in locked
