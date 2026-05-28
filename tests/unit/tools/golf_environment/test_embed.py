"""Smoke tests for the golf_environment EmbeddableTool adapter.

These tests are strictly headless — PyQt6 is checked via
``pytest.importorskip`` so the suite runs in headless CI without a
display or Qt installation.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6", reason="PyQt6 is required for golf_environment")

from src.shared.python.launcher_embed import (  # noqa: E402
    EmbedCapabilities,
    EmbeddableTool,
)


@pytest.fixture()
def adapter():
    """Return a fresh _EmbedAdapter instance."""
    from src.tools.golf_environment.gui import _EmbedAdapter

    return _EmbedAdapter()


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_tool_id_is_stable() -> None:
    from src.tools.golf_environment.gui import _EmbedAdapter

    assert _EmbedAdapter.tool_id == "golf_environment"
    assert _EmbedAdapter().tool_id == "golf_environment"


@pytest.mark.unit
def test_adapter_satisfies_embeddable_protocol(adapter) -> None:
    assert isinstance(adapter, EmbeddableTool)


# ---------------------------------------------------------------------------
# embed_capabilities
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_embed_capabilities_returns_expected_shape(adapter) -> None:
    caps = adapter.embed_capabilities()

    assert isinstance(caps, EmbedCapabilities)
    assert caps.supports_embedded is True
    assert caps.requires_separate_qapplication is False
    assert caps.min_size[0] > 0
    assert caps.min_size[1] > 0


@pytest.mark.unit
def test_embed_capabilities_stable_across_calls(adapter) -> None:
    assert adapter.embed_capabilities() == adapter.embed_capabilities()


# ---------------------------------------------------------------------------
# cleanup / is_dirty
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cleanup_before_widget_is_safe(adapter) -> None:
    adapter.cleanup()
    assert adapter._widget is None


@pytest.mark.unit
def test_cleanup_is_idempotent(adapter) -> None:
    adapter.cleanup()
    adapter.cleanup()


@pytest.mark.unit
def test_is_dirty_always_false(adapter) -> None:
    assert adapter.is_dirty() is False


# ---------------------------------------------------------------------------
# Registration side-effect
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_gui_module_registers_adapter_on_import() -> None:
    from src.shared.python.launcher_embed import get_embeddable_tool

    import src.tools.golf_environment.gui  # noqa: F401

    registered = get_embeddable_tool("golf_environment")
    assert registered is not None
    assert registered.tool_id == "golf_environment"
    assert isinstance(registered, EmbeddableTool)
