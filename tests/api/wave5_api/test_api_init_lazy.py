"""Tests for src/api/__init__.py lazy-loading."""

from __future__ import annotations

import importlib

import pytest

pytestmark = pytest.mark.unit


def test_versioning_pre_loaded() -> None:
    api_mod = importlib.import_module("src.api")
    # Pre-populated by _preload_versioning
    assert hasattr(api_mod, "versioning")
    assert api_mod.versioning.get_app_version


def test_unknown_attribute_raises() -> None:
    api_mod = importlib.import_module("src.api")
    with pytest.raises(AttributeError):
        _ = api_mod.does_not_exist  # type: ignore[attr-defined]


def test_attribute_name_must_be_str() -> None:
    api_mod = importlib.import_module("src.api")
    with pytest.raises(TypeError):
        api_mod.__getattr__(123)  # type: ignore[arg-type]


def test_lazy_attrs_mapping_consistent() -> None:
    api_mod = importlib.import_module("src.api")
    for name in api_mod._LAZY_ATTRS:
        assert name in api_mod.__all__
