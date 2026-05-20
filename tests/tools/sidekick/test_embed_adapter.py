"""Unit tests for the Sidekick embed adapter.

Covers :mod:`src.tools.sidekick._embed_adapter` and the package-level
registration side-effect in :mod:`src.tools.sidekick`.

These tests stay strictly headless — PyQt6 and the heavy AI assistant
panel are mocked via :func:`unittest.mock.patch.dict` so the tests run
in <30s on CI machines without a display.
"""

from __future__ import annotations

import importlib
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from src.shared.python.launcher_embed import (
    EmbedCapabilities,
    EmbeddableTool,
    get_embeddable_tool,
)
from src.tools.sidekick._embed_adapter import _SidekickEmbedAdapter


# ---------------------------------------------------------------------------
# embed_capabilities
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_tool_id_is_sidekick() -> None:
    assert _SidekickEmbedAdapter.tool_id == "sidekick"
    assert _SidekickEmbedAdapter().tool_id == "sidekick"


@pytest.mark.unit
def test_embed_capabilities_returns_expected_shape() -> None:
    caps = _SidekickEmbedAdapter().embed_capabilities()

    assert isinstance(caps, EmbedCapabilities)
    assert caps.supports_embedded is True
    assert caps.prefers_dock is True
    assert caps.min_size == (360, 480)
    assert caps.requires_separate_qapplication is False


@pytest.mark.unit
def test_embed_capabilities_is_stable_across_calls() -> None:
    adapter = _SidekickEmbedAdapter()
    assert adapter.embed_capabilities() == adapter.embed_capabilities()


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_adapter_satisfies_embeddable_protocol() -> None:
    adapter = _SidekickEmbedAdapter()
    assert isinstance(adapter, EmbeddableTool)


# ---------------------------------------------------------------------------
# create_main_widget
# ---------------------------------------------------------------------------


def _install_fake_assistant_panel_module(monkeypatch, panel_cls) -> None:
    """Inject a fake ``src.shared.python.ai.gui.assistant_panel`` module."""
    fake_mod = types.ModuleType("src.shared.python.ai.gui.assistant_panel")
    fake_mod.AIAssistantPanel = panel_cls
    monkeypatch.setitem(
        sys.modules, "src.shared.python.ai.gui.assistant_panel", fake_mod
    )


@pytest.mark.unit
def test_create_main_widget_instantiates_panel_with_parent(monkeypatch) -> None:
    panel_cls = MagicMock(name="AIAssistantPanel")
    panel_cls.return_value = MagicMock(name="panel_instance")
    _install_fake_assistant_panel_module(monkeypatch, panel_cls)

    adapter = _SidekickEmbedAdapter()
    parent = object()
    widget = adapter.create_main_widget(parent)

    panel_cls.assert_called_once_with(parent)
    assert widget is panel_cls.return_value


@pytest.mark.unit
def test_create_main_widget_tracks_widgets(monkeypatch) -> None:
    panel_cls = MagicMock()
    panel_cls.side_effect = lambda parent: MagicMock(name=f"panel-{id(parent)}")
    _install_fake_assistant_panel_module(monkeypatch, panel_cls)

    adapter = _SidekickEmbedAdapter()
    w1 = adapter.create_main_widget(None)
    w2 = adapter.create_main_widget(None)

    assert adapter._widgets == [w1, w2]


@pytest.mark.unit
def test_create_main_widget_propagates_import_error(monkeypatch) -> None:
    # Ensure no cached real module.
    monkeypatch.setitem(
        sys.modules,
        "src.shared.python.ai.gui.assistant_panel",
        None,  # forces ImportError on next import
    )

    adapter = _SidekickEmbedAdapter()
    with pytest.raises(ImportError):
        adapter.create_main_widget(None)


# ---------------------------------------------------------------------------
# cleanup
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cleanup_calls_delete_later_on_widgets(monkeypatch) -> None:
    panel_cls = MagicMock()
    panel_cls.side_effect = lambda parent: MagicMock(spec=["deleteLater"])
    _install_fake_assistant_panel_module(monkeypatch, panel_cls)

    adapter = _SidekickEmbedAdapter()
    w1 = adapter.create_main_widget(None)
    w2 = adapter.create_main_widget(None)

    adapter.cleanup()

    w1.deleteLater.assert_called_once_with()
    w2.deleteLater.assert_called_once_with()
    assert adapter._widgets == []


@pytest.mark.unit
def test_cleanup_is_idempotent() -> None:
    adapter = _SidekickEmbedAdapter()
    adapter.cleanup()
    adapter.cleanup()
    assert adapter._widgets == []


@pytest.mark.unit
def test_cleanup_skips_widgets_without_delete_later() -> None:
    adapter = _SidekickEmbedAdapter()
    # Widget exposes no deleteLater attribute at all.
    adapter._widgets = [object()]
    adapter.cleanup()
    assert adapter._widgets == []


@pytest.mark.unit
def test_cleanup_skips_non_callable_delete_later() -> None:
    adapter = _SidekickEmbedAdapter()
    widget = MagicMock()
    widget.deleteLater = "not-callable"
    adapter._widgets = [widget]
    adapter.cleanup()
    assert adapter._widgets == []


@pytest.mark.unit
def test_cleanup_swallows_exceptions_and_logs() -> None:
    adapter = _SidekickEmbedAdapter()
    bad = MagicMock()
    bad.deleteLater.side_effect = RuntimeError("boom")
    good = MagicMock()
    adapter._widgets = [bad, good]

    with patch("src.tools.sidekick._embed_adapter.logger") as mock_logger:
        adapter.cleanup()  # must not raise

    mock_logger.exception.assert_called_once()
    # Even when one widget raises, the loop continues and the second
    # widget's deleter is invoked.
    good.deleteLater.assert_called_once_with()
    assert adapter._widgets == []


# ---------------------------------------------------------------------------
# is_dirty
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_is_dirty_always_false() -> None:
    adapter = _SidekickEmbedAdapter()
    assert adapter.is_dirty() is False

    # State changes (handing out widgets) must not flip is_dirty.
    adapter._widgets.append(MagicMock())
    assert adapter is not None and adapter.is_dirty() is False


# ---------------------------------------------------------------------------
# Package-level registration
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_package_registers_adapter_on_import() -> None:
    # Importing the package elsewhere in the test session should have
    # already registered the adapter. Re-import to be defensive.
    import src.tools.sidekick  # noqa: F401

    registered = get_embeddable_tool("sidekick")
    assert registered is not None
    assert registered.tool_id == "sidekick"
    assert isinstance(registered, _SidekickEmbedAdapter)


@pytest.mark.unit
def test_package_registration_is_idempotent() -> None:
    """Reloading the package must not raise or replace the adapter."""
    import src.tools.sidekick as pkg

    first = get_embeddable_tool("sidekick")
    importlib.reload(pkg)
    second = get_embeddable_tool("sidekick")

    # ``get_embeddable_tool`` returns the originally registered instance
    # because the guard in ``__init__`` skips re-registration.
    assert second is first


@pytest.mark.unit
def test_package_all_is_empty() -> None:
    import src.tools.sidekick as pkg

    assert pkg.__all__ == []
