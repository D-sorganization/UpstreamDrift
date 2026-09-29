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
    EMBEDDABLE_TOOL_REGISTRY,
    EmbedCapabilities,
    EmbeddableTool,
    get_embeddable_tool,
)
from src.tools.sidekick._embed_adapter import _SidekickEmbedAdapter

_SIDEKICK_PACKAGE = "src.tools.sidekick"

# ---------------------------------------------------------------------------
# Registry isolation (issue #9168)
# ---------------------------------------------------------------------------


def _sidekick_module_names() -> list[str]:
    """Return every currently-imported ``src.tools.sidekick*`` module name."""
    prefix = f"{_SIDEKICK_PACKAGE}."
    return [
        name
        for name in list(sys.modules)
        if name == _SIDEKICK_PACKAGE or name.startswith(prefix)
    ]


@pytest.fixture(autouse=True)
def _isolate_sidekick_registration():
    """Snapshot and restore the registry *and* the sidekick module cache.

    ``src.tools.sidekick`` registers its adapter as an *import* side
    effect, so the registration runs at most once per process. Two of the
    tests below want opposite preconditions — "registers on import" needs
    the entry absent, "idempotent" needs it already present — and before
    #9168 each simply inherited whatever ambient state the session had
    produced. Under ``pytest-xdist`` that state depends on whether some
    other test in the same worker imported the package first, which is
    pure scheduling luck, so the pair failed as mirror images:
    ``assert None is not None`` and ``assert <adapter object> is None``.

    Both halves of the state have to be restored. Putting the registry
    back is not enough on its own: if the package stays cached in
    ``sys.modules`` the import side effect never re-runs, so a later test
    still sees an empty registry. Copying the mapping and the module
    entries — rather than rebinding names — keeps each test's mutations
    from escaping into the rest of the worker's session.
    """
    registry_snapshot = dict(EMBEDDABLE_TOOL_REGISTRY)
    module_snapshot = {name: sys.modules[name] for name in _sidekick_module_names()}
    try:
        yield
    finally:
        EMBEDDABLE_TOOL_REGISTRY.clear()
        EMBEDDABLE_TOOL_REGISTRY.update(registry_snapshot)
        for name in _sidekick_module_names():
            del sys.modules[name]
        sys.modules.update(module_snapshot)


def _fresh_import_sidekick():
    """Drop the sidekick package from ``sys.modules`` and import it again.

    Forces the module-level registration side effect to actually execute
    instead of being short-circuited by an ambient cached import. Returns
    the freshly imported package.
    """
    for name in _sidekick_module_names():
        del sys.modules[name]
    return importlib.import_module(_SIDEKICK_PACKAGE)


def _fresh_adapter_class() -> type:
    """Return the ``_SidekickEmbedAdapter`` class from the live module.

    A fresh import rebinds the class object, so identity checks must use
    the class the *current* module cache holds, not the one this test
    module imported at collection time.
    """
    module = importlib.import_module(f"{_SIDEKICK_PACKAGE}._embed_adapter")
    return module._SidekickEmbedAdapter


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
    # Resolve class and patch target from the *same* live module object.
    # A dotted-string patch target would re-resolve through ``sys.modules``,
    # which another suite may have swapped for a freshly imported module
    # (``tests/unit/launcher_embed/test_sidekick_contract.py`` evicts the
    # sidekick modules). Patching the module the class actually belongs to
    # makes this independent of import order. See issue #9168.
    module = importlib.import_module(f"{_SIDEKICK_PACKAGE}._embed_adapter")
    adapter = module._SidekickEmbedAdapter()
    bad = MagicMock()
    bad.deleteLater.side_effect = RuntimeError("boom")
    good = MagicMock()
    adapter._widgets = [bad, good]

    with patch.object(module, "logger") as mock_logger:
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
    """Importing the package registers the adapter.

    Both preconditions are established explicitly rather than inherited
    from whatever import order the session produced: the registry entry
    is dropped, and the package is evicted from ``sys.modules`` so the
    import genuinely re-executes the registration.
    """
    EMBEDDABLE_TOOL_REGISTRY.pop("sidekick", None)
    assert get_embeddable_tool("sidekick") is None

    _fresh_import_sidekick()

    registered = get_embeddable_tool("sidekick")
    assert registered is not None
    assert registered.tool_id == "sidekick"
    assert isinstance(registered, _fresh_adapter_class())


@pytest.mark.unit
def test_package_registration_is_idempotent() -> None:
    """Re-importing the package must not raise or replace the adapter."""
    # Explicitly establish the "already registered" precondition instead
    # of assuming an earlier test in this worker supplied it.
    EMBEDDABLE_TOOL_REGISTRY.pop("sidekick", None)
    _fresh_import_sidekick()
    first = get_embeddable_tool("sidekick")
    assert first is not None

    _fresh_import_sidekick()
    second = get_embeddable_tool("sidekick")

    # ``get_embeddable_tool`` returns the originally registered instance
    # because the guard in ``__init__`` skips re-registration.
    assert second is first


@pytest.mark.unit
def test_package_registration_recovers_from_cleared_registry() -> None:
    """A cleared registry must be repopulated by re-importing the package.

    Regression guard for #9168: the adapter is registered by an import
    side effect, so once another suite clears the registry the entry only
    returns if the package body runs again. Asserting that recovery path
    directly is what makes this file independent of import order.
    """
    _fresh_import_sidekick()
    EMBEDDABLE_TOOL_REGISTRY.clear()
    assert get_embeddable_tool("sidekick") is None

    _fresh_import_sidekick()

    assert isinstance(get_embeddable_tool("sidekick"), _fresh_adapter_class())


@pytest.mark.unit
def test_package_all_is_empty() -> None:
    pkg = _fresh_import_sidekick()

    assert pkg.__all__ == []
