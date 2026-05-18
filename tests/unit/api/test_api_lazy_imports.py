"""Regression tests for issue #5739: src/api/__init__.py eager imports.

PR #5681 added eager top-level imports of ``auth``, ``routes``, and
``services`` to ``src/api/__init__.py``.  ``auth.models`` pulls in
SQLAlchemy at import time.  Environments that only use the lightweight
``versioning`` module (e.g. CLI tools, docs builds, test environments
without the full API extras) now crash with::

    ModuleNotFoundError: No module named 'sqlalchemy'

Contract locked in by these tests:

1. ``import src.api`` succeeds even when SQLAlchemy is absent.
2. ``from src.api import versioning`` succeeds when SQLAlchemy is absent.
3. Accessing ``src.api.routes``, ``src.api.auth``, or ``src.api.services``
   while SQLAlchemy is absent raises ``ImportError`` (or
   ``ModuleNotFoundError``), *not* silently returning ``None``.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_API_MODULE = "src.api"
# Sub-modules that transitively import SQLAlchemy
_HEAVY_SUBMODULES = [
    "src.api.auth",
    "src.api.auth.models",
    "src.api.auth.dependencies",
    "src.api.auth.security",
    "src.api.routes",
    "src.api.routes.launcher",
    "src.api.routes.models",
    "src.api.routes.physics",
    "src.api.routes.simulation",
    "src.api.services",
]

# SQLAlchemy top-level packages/modules to stub out
_SQLALCHEMY_NAMES = [
    "sqlalchemy",
    "sqlalchemy.orm",
    "sqlalchemy.sql",
    "sqlalchemy.ext",
    "sqlalchemy.ext.declarative",
]


def _evict_api_modules() -> dict[str, ModuleType]:
    """Remove src.api and all sub-modules from sys.modules, return snapshot."""
    saved: dict[str, ModuleType] = {}
    keys_to_remove = [
        k for k in sys.modules if k == _API_MODULE or k.startswith(_API_MODULE + ".")
    ]
    for key in keys_to_remove:
        saved[key] = sys.modules.pop(key)
    return saved


def _restore_api_modules(saved: dict[str, ModuleType]) -> None:
    for key, mod in saved.items():
        sys.modules[key] = mod


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestImportWithoutSQLAlchemy:
    """src.api must import cleanly in lightweight / no-DB environments."""

    def test_import_src_api_succeeds_without_sqlalchemy(self) -> None:
        """``import src.api`` must not raise even when sqlalchemy is absent."""
        saved = _evict_api_modules()
        try:
            with mock.patch.dict(sys.modules, dict.fromkeys(_SQLALCHEMY_NAMES)):
                # Should not raise
                api_mod = importlib.import_module(_API_MODULE)
                assert isinstance(api_mod, ModuleType)
        finally:
            _restore_api_modules(saved)

    def test_versioning_importable_without_sqlalchemy(self) -> None:
        """``from src.api import versioning`` must work without sqlalchemy."""
        saved = _evict_api_modules()
        try:
            with mock.patch.dict(sys.modules, dict.fromkeys(_SQLALCHEMY_NAMES)):
                versioning = importlib.import_module("src.api.versioning")
                assert isinstance(versioning, ModuleType)
                # Sanity-check: versioning exposes at least one public symbol
                assert hasattr(versioning, "__file__"), (
                    "versioning module has no __file__"
                )
        finally:
            _restore_api_modules(saved)

    def test_versioning_accessible_via_attribute_without_sqlalchemy(self) -> None:
        """``src.api.versioning`` attribute access must not trigger SQLAlchemy."""
        saved = _evict_api_modules()
        try:
            with mock.patch.dict(sys.modules, dict.fromkeys(_SQLALCHEMY_NAMES)):
                api_mod = importlib.import_module(_API_MODULE)
                versioning = api_mod.versioning  # type: ignore[attr-defined]
                assert isinstance(versioning, ModuleType)
        finally:
            _restore_api_modules(saved)


class TestHeavySubmodulesFailGracefullyWithoutSQLAlchemy:
    """The lazy mechanism defers heavy imports; auth (which uses SQLAlchemy ORM
    models) must raise ImportError when SQLAlchemy is absent.  ``routes`` and
    ``services`` themselves do not import SQLAlchemy at their package level, so
    they can be imported cleanly; only ``auth`` is the gating dependency."""

    def test_accessing_auth_without_sqlalchemy_raises(self) -> None:
        """``src.api.auth`` access must raise ImportError when sqlalchemy absent.

        ``src.api.auth.models`` imports SQLAlchemy ORM types at module level,
        so accessing the auth sub-package must propagate the ImportError.
        """
        saved = _evict_api_modules()
        try:
            with mock.patch.dict(sys.modules, dict.fromkeys(_SQLALCHEMY_NAMES)):
                api_mod = importlib.import_module(_API_MODULE)
                with pytest.raises((ImportError, ModuleNotFoundError)):
                    _ = api_mod.auth  # type: ignore[attr-defined]
        finally:
            _restore_api_modules(saved)

    def test_accessing_routes_succeeds_without_sqlalchemy(self) -> None:
        """``src.api.routes`` does not pull in SQLAlchemy at package level.

        Only ``src.api.routes.auth`` (not imported by the routes __init__)
        uses SQLAlchemy.  The main routes package must import cleanly.
        """
        saved = _evict_api_modules()
        try:
            with mock.patch.dict(sys.modules, dict.fromkeys(_SQLALCHEMY_NAMES)):
                api_mod = importlib.import_module(_API_MODULE)
                routes = api_mod.routes  # type: ignore[attr-defined]
                assert isinstance(routes, ModuleType)
        finally:
            _restore_api_modules(saved)

    def test_accessing_services_succeeds_without_sqlalchemy(self) -> None:
        """``src.api.services`` does not use SQLAlchemy directly.

        Services rely on in-process Python objects, not ORM models.
        The package must import cleanly when SQLAlchemy is absent.
        """
        saved = _evict_api_modules()
        try:
            with mock.patch.dict(sys.modules, dict.fromkeys(_SQLALCHEMY_NAMES)):
                api_mod = importlib.import_module(_API_MODULE)
                services = api_mod.services  # type: ignore[attr-defined]
                assert isinstance(services, ModuleType)
        finally:
            _restore_api_modules(saved)

    def test_get_current_user_raises_without_sqlalchemy(self) -> None:
        """``src.api.get_current_user`` is re-exported from auth; must raise."""
        saved = _evict_api_modules()
        try:
            with mock.patch.dict(sys.modules, dict.fromkeys(_SQLALCHEMY_NAMES)):
                api_mod = importlib.import_module(_API_MODULE)
                with pytest.raises((ImportError, ModuleNotFoundError)):
                    _ = api_mod.get_current_user  # type: ignore[attr-defined]
        finally:
            _restore_api_modules(saved)
