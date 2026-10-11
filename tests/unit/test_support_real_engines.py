"""``real_module`` skips on engine test doubles instead of running against them (#12204)."""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest

from tests.support.real_engines import is_test_double, real_module

pytestmark = pytest.mark.unit


def test_magicmock_is_a_test_double() -> None:
    assert is_test_double(MagicMock())


def test_spec_less_stub_module_is_a_test_double() -> None:
    assert is_test_double(types.ModuleType("pydrake"))


def test_real_module_is_not_a_test_double() -> None:
    assert not is_test_double(pytest)


def test_real_module_skips_on_conftest_style_mock() -> None:
    mock = MagicMock()
    with (
        patch.dict("sys.modules", {"pydrake": mock, "pydrake.all": mock}),
        pytest.raises(pytest.skip.Exception, match="test double"),
    ):
        real_module("pydrake.all")


def test_real_module_skips_when_submodule_of_mocked_root() -> None:
    """A real-looking submodule under a mocked root package is not real either."""
    stub = types.ModuleType("fake_engine.sub")
    stub.__file__ = "fake_engine/sub.py"
    with (
        patch.dict(
            "sys.modules", {"fake_engine": MagicMock(), "fake_engine.sub": stub}
        ),
        pytest.raises(pytest.skip.Exception, match="test double"),
    ):
        real_module("fake_engine.sub")


def test_real_module_skips_when_not_installed() -> None:
    with pytest.raises(pytest.skip.Exception):
        real_module("definitely_not_an_installed_engine_12204")


def test_real_module_returns_installed_package() -> None:
    assert real_module("json") is __import__("json")


def test_real_module_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="module name"):
        real_module("")
