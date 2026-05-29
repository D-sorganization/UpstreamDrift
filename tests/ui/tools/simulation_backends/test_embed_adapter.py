"""Contract tests for the Simulation Backends embed adapter.

Verifies that ``_EmbedAdapter`` satisfies the :class:`EmbeddableTool`
protocol, registers itself under the manifest id ``simulation_backends``,
and builds a real ``QWidget`` on demand. The adapter module is PyQt6-free,
so the protocol/id assertions do not need a ``QApplication``; only
``create_main_widget`` does.
"""

from __future__ import annotations

import importlib

import pytest

from src.shared.python.launcher_embed import (
    EmbedCapabilities,
    EmbeddableTool,
    get_embeddable_tool,
    is_embeddable,
)
from src.tools.simulation_backends_launcher import _embed_adapter

pytestmark = [pytest.mark.unit]


def _fresh_adapter() -> object:
    """Return a new ``_EmbedAdapter`` instance."""
    return _embed_adapter._EmbedAdapter()


def test_adapter_satisfies_protocol() -> None:
    """The adapter is a structural ``EmbeddableTool``."""
    assert isinstance(_fresh_adapter(), EmbeddableTool)


def test_tool_id_is_simulation_backends() -> None:
    """The registry id matches the launcher manifest entry."""
    adapter = _fresh_adapter()
    assert isinstance(adapter.tool_id, str)
    assert adapter.tool_id == "simulation_backends"


def test_embed_capabilities_are_valid() -> None:
    """``embed_capabilities`` returns a valid, embeddable capability set."""
    caps = _fresh_adapter().embed_capabilities()
    assert isinstance(caps, EmbedCapabilities)
    assert caps.supports_embedded is True
    assert caps.requires_separate_qapplication is False
    assert isinstance(caps.min_size, tuple)
    assert len(caps.min_size) == 2
    assert all(isinstance(dim, int) and dim > 0 for dim in caps.min_size)


def test_is_dirty_defaults_false() -> None:
    """The tool holds no persistent state, so it is never dirty."""
    assert _fresh_adapter().is_dirty() is False


def test_cleanup_is_idempotent() -> None:
    """``cleanup`` is safe to call repeatedly."""
    adapter = _fresh_adapter()
    adapter.cleanup()
    adapter.cleanup()


def test_import_registers_tool() -> None:
    """Importing the adapter module self-registers the tile.

    The autouse fixture clears the registry, so the module is reloaded here
    to re-run its module-level ``register_embeddable_tool`` call.
    """
    importlib.reload(_embed_adapter)
    tool = get_embeddable_tool("simulation_backends")
    assert tool is not None
    assert isinstance(tool, EmbeddableTool)
    assert is_embeddable("simulation_backends")


def test_create_main_widget_returns_qwidget(qapp) -> None:  # noqa: ANN001
    """``create_main_widget`` builds a real ``QWidget`` (needs a QApplication)."""
    from PyQt6 import QtWidgets

    adapter = _fresh_adapter()
    widget = adapter.create_main_widget(None)
    try:
        assert isinstance(widget, QtWidgets.QWidget)
    finally:
        widget.deleteLater()
