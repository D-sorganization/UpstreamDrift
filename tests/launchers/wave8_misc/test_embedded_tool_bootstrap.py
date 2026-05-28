"""Tests for src.launchers.embedded_tool_bootstrap.

The bootstrap module imports a hard-coded list of adapter modules at
runtime to trigger their self-registration with the embeddable tool
registry. These tests verify:

* Idempotency: second call returns cached results without re-importing.
* Path injection: vendor/ud-tools/src is added to ``sys.path``.
* Graceful degradation: failing modules log a warning but do not abort.
* State reset for tests.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from src.launchers import embedded_tool_bootstrap as bootstrap


@pytest.fixture(autouse=True)
def _reset_bootstrap_state():
    """Reset module-level state before and after each test."""
    bootstrap.reset_bootstrap_state()
    yield
    bootstrap.reset_bootstrap_state()


def test_reset_clears_state() -> None:
    bootstrap._registered_tools = ["foo", "bar"]
    bootstrap._bootstrap_complete = True
    bootstrap.reset_bootstrap_state()
    assert bootstrap._registered_tools == []
    assert bootstrap._bootstrap_complete is False


def test_get_bootstrapped_tools_returns_copy() -> None:
    bootstrap._registered_tools = ["a", "b"]
    result = bootstrap.get_bootstrapped_tools()
    assert result == ["a", "b"]
    # Mutating the returned list should not change internal state
    result.append("c")
    assert bootstrap._registered_tools == ["a", "b"]


import builtins as _builtins  # noqa: E402

_ADAPTER_TOKEN = "_embed_adapter"
_REAL_IMPORT = _builtins.__import__


def _make_filtered_import(decide):
    """Build an __import__ replacement that delegates to the real one
    for everything except adapter module names, which go through *decide*.
    """

    def _filtered(name, globals=None, locals=None, fromlist=(), level=0):
        if _ADAPTER_TOKEN in name or "pose_studio.gui" in name:
            return decide(name)
        return _REAL_IMPORT(name, globals, locals, fromlist, level)

    return _filtered


def test_bootstrap_handles_import_errors() -> None:
    """When all adapter modules fail to import, return an empty list."""

    def _fail(name):
        raise ImportError(f"forced for {name}")

    with patch.object(_builtins, "__import__", _make_filtered_import(_fail)):
        result = bootstrap.bootstrap_embeddable_tools()
    assert result == []
    assert bootstrap._bootstrap_complete is True


def test_bootstrap_handles_generic_errors() -> None:
    """Non-ImportError exceptions during import are also swallowed."""

    def _boom(name):
        raise RuntimeError(f"forced runtime for {name}")

    with patch.object(_builtins, "__import__", _make_filtered_import(_boom)):
        result = bootstrap.bootstrap_embeddable_tools()
    assert result == []
    assert bootstrap._bootstrap_complete is True


def test_bootstrap_is_idempotent() -> None:
    """Calling bootstrap twice should not re-run adapter imports."""
    calls = {"n": 0}

    def _counting(name):
        calls["n"] += 1
        raise ImportError("forced")

    with patch.object(_builtins, "__import__", _make_filtered_import(_counting)):
        first = bootstrap.bootstrap_embeddable_tools()
        first_count = calls["n"]
        second = bootstrap.bootstrap_embeddable_tools()
    assert second == first
    assert calls["n"] == first_count


def test_bootstrap_adds_vendor_path_to_sys_path() -> None:
    """sys.path should gain the vendor/ud-tools/src directory."""
    from pathlib import Path

    vendor_src = str(
        Path(bootstrap.__file__).resolve().parent.parent.parent.parent
        / "vendor"
        / "ud-tools"
        / "src"
    )
    original_path = list(sys.path)
    try:
        sys.path[:] = [p for p in sys.path if p != vendor_src]

        def _noop(name):
            raise ImportError("noop")

        with patch.object(_builtins, "__import__", _make_filtered_import(_noop)):
            bootstrap.bootstrap_embeddable_tools()
        assert vendor_src in sys.path
    finally:
        sys.path[:] = original_path


def test_bootstrap_records_successfully_imported_tools() -> None:
    """Successful imports get their tool_id extracted and recorded."""

    def _selective(name):
        if name == "src.tools.model_explorer._embed_adapter":
            return object()  # success
        raise ImportError(f"forced for {name}")

    with patch.object(_builtins, "__import__", _make_filtered_import(_selective)):
        result = bootstrap.bootstrap_embeddable_tools()
    assert "model_explorer" in result
