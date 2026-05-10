"""Tests for the Drake dashboard embed adapter.

Covers:
- Protocol conformance against
  :class:`src.shared.python.launcher_embed.EmbeddableTool`.
- :meth:`embed_capabilities` matches the spec for #4998.
- :meth:`create_main_widget` returns a real ``QWidget`` (skipped when
  the ``pydrake`` wheel is not installed).
- Importing the host package produces the registry side-effect.

Part of Subtask 5 / #4998 of EPIC #4993.
"""

from __future__ import annotations

import importlib
import sys

import pytest

# The dashboard pulls in PyQt6 and (for widget construction) the
# ``pydrake`` wheel. Skip the whole module rather than failing
# collection on hosts where PyQt6 is missing.
PyQt6 = pytest.importorskip("PyQt6")

from PyQt6 import QtWidgets  # noqa: E402

from src.shared.python.launcher_embed import (  # noqa: E402
    EmbedCapabilities,
    EmbeddableTool,
)

_DRAKE_PKG_NAME = "src.engines.physics_engines.drake.python.src"
_ADAPTER_MOD_NAME = f"{_DRAKE_PKG_NAME}._embed_adapter"


def _import_drake_pkg():
    """Import the Drake engine ``src`` package and return it."""
    try:
        return importlib.import_module(_DRAKE_PKG_NAME)
    except ImportError as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"Drake engine package not importable: {exc}")


def _import_adapter_module():
    """Import (and return) the adapter module, skipping cleanly if it
    cannot be imported (e.g. when PyQt6 transitively pulls in something
    unavailable on this host)."""
    try:
        return importlib.import_module(_ADAPTER_MOD_NAME)
    except ImportError as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"Drake embed adapter not importable: {exc}")


@pytest.fixture
def adapter():
    """Construct a fresh adapter instance for each test."""
    _import_drake_pkg()
    adapter_mod = _import_adapter_module()
    yield adapter_mod._DrakeDashboardEmbedAdapter()


# --- Protocol conformance ------------------------------------------------


@pytest.mark.unit
def test_adapter_satisfies_embeddable_tool_protocol(adapter) -> None:
    """The adapter is a structural :class:`EmbeddableTool`."""
    assert isinstance(adapter, EmbeddableTool)
    assert adapter.tool_id == "drake_golf"


@pytest.mark.unit
def test_embed_capabilities_match_spec(adapter) -> None:
    """Capabilities match the values documented for Subtask 5 / #4998."""
    caps = adapter.embed_capabilities()
    assert isinstance(caps, EmbedCapabilities)
    assert caps.supports_embedded is True
    assert caps.prefers_dock is False
    assert caps.min_size == (1000, 700)
    assert caps.requires_separate_qapplication is False


@pytest.mark.unit
def test_is_dirty_default_is_false(adapter) -> None:
    """The dashboard does not track an implicit dirty state."""
    assert adapter.is_dirty() is False


@pytest.mark.unit
def test_cleanup_is_idempotent(adapter) -> None:
    """``cleanup`` may be called repeatedly without raising."""
    adapter.cleanup()
    adapter.cleanup()


# --- create_main_widget returns a real QWidget -------------------------


@pytest.mark.unit
def test_create_main_widget_returns_qwidget(qapp, adapter) -> None:  # noqa: ANN001
    """``create_main_widget(None)`` returns a ``QWidget``.

    Skipped when the ``pydrake`` wheel is not installed (or its native
    extensions cannot be loaded on this host) because the dashboard's
    :class:`MainWidget` constructs a Drake simulator on initialization.
    """
    pytest.importorskip("pydrake")
    # ``pydrake`` is a namespace package, so the importorskip above
    # succeeds even when the native wheel failed to load. Probe the
    # specific symbol the dashboard depends on.
    try:
        from pydrake.all import MeshcatParams  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"pydrake native wheel not loadable: {exc}")

    widget = adapter.create_main_widget(None)
    try:
        assert isinstance(widget, QtWidgets.QWidget)
    finally:
        adapter.cleanup()
        widget.deleteLater()


# --- Registry side-effect on import -------------------------------------


@pytest.mark.unit
def test_drake_package_import_registers_adapter() -> None:
    """Importing the Drake engine package registers the adapter."""
    from src.shared.python.launcher_embed import (
        EMBEDDABLE_TOOL_REGISTRY,
        get_embeddable_tool,
        unregister_embeddable_tool,
    )

    # Clear any prior registration so we observe the import-time effect
    # cleanly. Other tools in the registry are left untouched.
    if "drake_golf" in EMBEDDABLE_TOOL_REGISTRY:
        unregister_embeddable_tool("drake_golf")

    # Re-import the package; ``__init__.py`` registers the adapter via
    # the guarded ``_embed_adapter`` import.
    sys.modules.pop(_DRAKE_PKG_NAME, None)
    sys.modules.pop(_ADAPTER_MOD_NAME, None)
    _import_drake_pkg()

    registered = get_embeddable_tool("drake_golf")
    if registered is None:  # pragma: no cover - environment-dependent
        pytest.skip(
            "drake_golf adapter not registered — likely PyQt6 unavailable "
            "and the contextlib.suppress(ImportError) guard fired."
        )
    assert registered.tool_id == "drake_golf"
    assert isinstance(registered, EmbeddableTool)
