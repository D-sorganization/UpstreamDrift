"""Tests for src.shared.python.data_io.import_utils (Issues #1949, #1744)."""

from __future__ import annotations

import sys

import pytest

from src.shared.python.data_io.import_utils import (
    check_minimum_version,
    check_optional_dependency,
    ensure_imports,
    get_module_version,
    import_from,
    lazy_import,
)

# ---------------------------------------------------------------------------
# ensure_imports
# ---------------------------------------------------------------------------


class TestEnsureImports:
    def test_available_module(self) -> None:
        result = ensure_imports("sys")
        assert result == {"sys": True}

    def test_unavailable_module(self) -> None:
        result = ensure_imports("_definitely_not_a_real_module_xyz")
        assert result == {"_definitely_not_a_real_module_xyz": False}

    def test_mixed_availability(self) -> None:
        result = ensure_imports("sys", "_no_such_module_abc")
        assert result["sys"] is True
        assert result["_no_such_module_abc"] is False

    def test_no_arguments_returns_empty(self) -> None:
        assert ensure_imports() == {}

    def test_multiple_available(self) -> None:
        result = ensure_imports("os", "sys", "math")
        assert all(result[m] is True for m in ("os", "sys", "math"))


# ---------------------------------------------------------------------------
# lazy_import
# ---------------------------------------------------------------------------


class TestLazyImport:
    def test_returns_module(self) -> None:
        mod = lazy_import("sys")
        assert mod is sys

    def test_missing_module_raises_import_error(self) -> None:
        with pytest.raises(ImportError):
            lazy_import("_no_such_module_xyz")


# ---------------------------------------------------------------------------
# import_from
# ---------------------------------------------------------------------------


class TestImportFrom:
    def test_single_name(self) -> None:
        path_cls = import_from("pathlib", "Path")
        from pathlib import Path

        assert path_cls is Path

    def test_multiple_names(self) -> None:
        result = import_from("os.path", "join", "exists")
        import os.path

        join_fn, exists_fn = result
        assert join_fn is os.path.join
        assert exists_fn is os.path.exists

    def test_missing_attribute_raises(self) -> None:
        with pytest.raises(AttributeError):
            import_from("sys", "_no_such_attr_xyz")


# ---------------------------------------------------------------------------
# check_optional_dependency
# ---------------------------------------------------------------------------


class TestCheckOptionalDependency:
    def test_available_returns_true(self) -> None:
        assert check_optional_dependency("sys") is True

    def test_missing_returns_false(self) -> None:
        assert check_optional_dependency("_no_such_module_xyz") is False

    def test_feature_name_does_not_raise(self) -> None:
        # Should log but not raise
        result = check_optional_dependency("_no_such_module", "My Feature")
        assert result is False


# ---------------------------------------------------------------------------
# get_module_version
# ---------------------------------------------------------------------------


class TestGetModuleVersion:
    def test_numpy_has_version(self) -> None:
        version = get_module_version("numpy")
        assert version is not None
        assert isinstance(version, str)
        assert "." in version

    def test_missing_module_returns_none(self) -> None:
        assert get_module_version("_no_such_module_xyz") is None


# ---------------------------------------------------------------------------
# check_minimum_version
# ---------------------------------------------------------------------------


class TestCheckMinimumVersion:
    def test_numpy_meets_ancient_minimum(self) -> None:
        assert check_minimum_version("numpy", "1.0.0") is True

    def test_impossible_minimum_fails(self) -> None:
        assert check_minimum_version("numpy", "9999.0.0") is False

    def test_missing_module_returns_false(self) -> None:
        assert check_minimum_version("_no_such_module", "1.0") is False
