"""Hygiene tests for cross-engine API version discipline.

Per issue #5917 (Category J: API Design), the canonical integration
surfaces ``realtime/``, ``pose_interchange/``, and ``launcher_embed/``
must expose ``__version__`` and ``SCHEMA_VERSION`` constants following
SemVer MAJOR.MINOR.PATCH so downstream consumers have a stable
contract for detecting breaking changes (ADR-0007, ADR-0012, ADR-0013).
"""

from __future__ import annotations

import importlib
import re

import pytest

SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")

PACKAGES = [
    "src.shared.python.realtime",
    "src.shared.python.pose_interchange",
    "src.shared.python.launcher_embed",
]


@pytest.mark.unit
@pytest.mark.parametrize("package", PACKAGES)
def test_package_exposes_version(package: str) -> None:
    """Every cross-engine API package exposes ``__version__``."""
    mod = importlib.import_module(package)
    assert hasattr(mod, "__version__"), f"{package} missing __version__"
    version = mod.__version__
    assert isinstance(version, str), f"{package}.__version__ must be str"
    assert SEMVER_RE.match(version), (
        f"{package}.__version__ = {version!r} is not SemVer MAJOR.MINOR.PATCH"
    )


@pytest.mark.unit
@pytest.mark.parametrize("package", PACKAGES)
def test_package_exposes_schema_version(package: str) -> None:
    """Every cross-engine API package exposes ``SCHEMA_VERSION``."""
    mod = importlib.import_module(package)
    assert hasattr(mod, "SCHEMA_VERSION"), f"{package} missing SCHEMA_VERSION"
    schema = mod.SCHEMA_VERSION
    assert isinstance(schema, str), f"{package}.SCHEMA_VERSION must be str"
    assert SEMVER_RE.match(schema), (
        f"{package}.SCHEMA_VERSION = {schema!r} is not SemVer"
    )


@pytest.mark.unit
@pytest.mark.parametrize("package", PACKAGES)
def test_version_constants_in_all(package: str) -> None:
    """``__version__`` and ``SCHEMA_VERSION`` are listed in ``__all__``."""
    mod = importlib.import_module(package)
    public = set(getattr(mod, "__all__", ()))
    assert "__version__" in public, f"{package}.__all__ missing __version__"
    assert "SCHEMA_VERSION" in public, f"{package}.__all__ missing SCHEMA_VERSION"
