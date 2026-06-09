"""API runtime dependencies must be core, not dev-only (issue #7125).

``src/api/database.py`` imports SQLAlchemy + Alembic, ``src/api/auth/security.py``
imports bcrypt + PyJWT, ``src/api/auth/models.py`` uses pydantic ``EmailStr``
(needs email-validator), and cryptography backs token/JWT crypto. A plain
``pip install upstream-drift`` (no ``[dev]`` extra) must be able to import and
start the API, so these belong in ``[project].dependencies`` and in
``requirements.lock`` — not only in the ``dev`` extra.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_LOCK = _REPO_ROOT / "requirements.lock"

# Distribution names (PEP 503 lowercase) of the API runtime dependencies.
_API_RUNTIME_DEPS = (
    "sqlalchemy",
    "alembic",
    "bcrypt",
    "pyjwt",
    "cryptography",
    "email-validator",
)


def _canonical(requirement: str) -> str:
    """Return the lowercase distribution name from a requirement string."""
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
    dev = pyproject["project"]["optional-dependencies"]["dev"]
    return {_canonical(dep) for dep in dev}


@pytest.mark.parametrize("dep", _API_RUNTIME_DEPS)
def test_api_runtime_dep_is_core(dep: str, core_dep_names: set[str]) -> None:
    assert dep in core_dep_names, (
        f"{dep} is imported at API runtime and must be a core dependency "
        f"in [project].dependencies (issue #7125)"
    )


@pytest.mark.parametrize("dep", _API_RUNTIME_DEPS)
def test_api_runtime_dep_not_dev_only(
    dep: str, dev_dep_names: set[str], core_dep_names: set[str]
) -> None:
    # It must not be *only* in dev. If it appears in dev at all it must also be
    # core; the cleanest state is core-only, which we assert here.
    if dep in dev_dep_names:
        pytest.fail(
            f"{dep} should be removed from the [dev] extra now that it is a "
            f"core API runtime dependency (issue #7125)"
        )
    assert dep in core_dep_names


@pytest.mark.parametrize("dep", _API_RUNTIME_DEPS)
def test_api_runtime_dep_is_locked(dep: str) -> None:
    lock_text = _LOCK.read_text(encoding="utf-8")
    locked = {
        _canonical(line)
        for line in lock_text.splitlines()
        if "==" in line and not line.lstrip().startswith("#")
    }
    assert dep in locked, (
        f"{dep} must be pinned in requirements.lock (regenerate via "
        f"`make sync-deps`) so the core install resolves it (issue #7125)"
    )
