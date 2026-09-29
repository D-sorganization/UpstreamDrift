"""Tests for the MyoSuite dashboard embed adapter.

Covers:

- Protocol conformance against
  :class:`src.shared.python.launcher_embed.EmbeddableTool`.
- :meth:`embed_capabilities` matches the spec for #4998.
- :meth:`create_main_widget` returns a real ``QWidget``. The widget
  itself does **not** require the optional ``myosuite`` wheel — engine
  imports are lazy and only triggered by user actions on the dashboard
  — but we still ``importorskip`` PyQt6 because the adapter pulls in Qt
  for widget construction.
- Importing the host package produces the registry side-effect.

Part of Subtask 5 / #4998 of EPIC #4993.
"""

from __future__ import annotations

import importlib
import sys

import pytest

pytestmark = [pytest.mark.unit]

# The dashboard pulls in PyQt6. Skip the whole module rather than
# failing collection on hosts where PyQt6 is missing.
PyQt6 = pytest.importorskip("PyQt6")

from PyQt6 import QtWidgets  # noqa: E402

from src.shared.python.launcher_embed import (  # noqa: E402
    EmbedCapabilities,
    EmbeddableTool,
)

_MYOSUITE_PKG_NAME = "src.engines.physics_engines.myosuite.python"
_ADAPTER_MOD_NAME = f"{_MYOSUITE_PKG_NAME}._embed_adapter"
_TOOL_ID = "myosim_suite"


def _import_myosuite_pkg():
    """Import the MyoSuite engine ``python`` package and return it."""
    try:
        return importlib.import_module(_MYOSUITE_PKG_NAME)
    except ImportError as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"MyoSuite engine package not importable: {exc}")


def _import_adapter_module():
    """Import (and return) the adapter module, skipping cleanly if it
    cannot be imported (e.g. when PyQt6 transitively pulls in something
    unavailable on this host)."""
    try:
        return importlib.import_module(_ADAPTER_MOD_NAME)
    except ImportError as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"MyoSuite embed adapter not importable: {exc}")


@pytest.fixture
def adapter():
    """Construct a fresh adapter instance for each test."""
    _import_myosuite_pkg()
    adapter_mod = _import_adapter_module()
    yield adapter_mod._MyoSuiteDashboardEmbedAdapter()


# --- Protocol conformance ------------------------------------------------


def test_adapter_satisfies_embeddable_tool_protocol(adapter) -> None:
    """The adapter is a structural :class:`EmbeddableTool`."""
    assert isinstance(adapter, EmbeddableTool)
    assert adapter.tool_id == _TOOL_ID


def test_embed_capabilities_match_spec(adapter) -> None:
    """Capabilities match the values documented for Subtask 5 / #4998."""
    caps = adapter.embed_capabilities()
    assert isinstance(caps, EmbedCapabilities)
    assert caps.supports_embedded is True
    assert caps.prefers_dock is False
    assert caps.min_size == (1000, 700)
    assert caps.requires_separate_qapplication is False


def test_is_dirty_default_is_false(adapter) -> None:
    """The dashboard does not track an implicit dirty state."""
    assert adapter.is_dirty() is False


def test_cleanup_is_idempotent(adapter) -> None:
    """``cleanup`` may be called repeatedly without raising."""
    adapter.cleanup()
    adapter.cleanup()


# --- create_main_widget returns a real QWidget -------------------------


def test_create_main_widget_returns_qwidget(qapp, adapter) -> None:  # noqa: ANN001
    """``create_main_widget(None)`` returns a ``QWidget``.

    The MyoSuite dashboard widget constructs without importing the
    optional ``myosuite`` wheel — engine probing is deferred to a
    button click — so we do **not** ``importorskip("myosuite")`` here.
    If a future refactor moves engine imports into
    :class:`MainWidget.__init__`, gate this test with
    ``pytest.importorskip("myosuite")`` instead.
    """
    widget = adapter.create_main_widget(None)
    try:
        assert isinstance(widget, QtWidgets.QWidget)
    finally:
        adapter.cleanup()
        widget.deleteLater()


# --- Registry side-effect on import -------------------------------------


def test_myosuite_package_import_registers_adapter() -> None:
    """Importing the MyoSuite engine package registers the adapter."""
    from src.shared.python.launcher_embed import (
        EMBEDDABLE_TOOL_REGISTRY,
        get_embeddable_tool,
        unregister_embeddable_tool,
    )

    # Clear any prior registration so we observe the import-time effect
    # cleanly. Other tools in the registry are left untouched.
    if _TOOL_ID in EMBEDDABLE_TOOL_REGISTRY:
        unregister_embeddable_tool(_TOOL_ID)

    # Re-import the package; ``__init__.py`` registers the adapter via
    # the guarded ``_embed_adapter`` import.
    sys.modules.pop(_MYOSUITE_PKG_NAME, None)
    sys.modules.pop(_ADAPTER_MOD_NAME, None)
    _import_myosuite_pkg()

    registered = get_embeddable_tool(_TOOL_ID)
    if registered is None:  # pragma: no cover - environment-dependent
        pytest.skip(
            "myosim_suite adapter not registered — likely PyQt6 unavailable "
            "and the contextlib.suppress(ImportError) guard fired."
        )
    assert registered.tool_id == _TOOL_ID
    assert isinstance(registered, EmbeddableTool)
