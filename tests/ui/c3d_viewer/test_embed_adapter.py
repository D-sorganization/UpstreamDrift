"""Tests for the C3D Viewer embed adapter.

Covers:
- Protocol conformance against
  :class:`src.shared.python.launcher_embed.EmbeddableTool`.
- :meth:`cleanup` releases all matplotlib figures.
- :meth:`create_main_widget` returns a ``QWidget``.
- Importing the host package produces the registry side-effect.
"""

from __future__ import annotations

import importlib
import sys

import pytest

# The viewer pulls in PyQt6, matplotlib, and the C3D reader chain.
# Skip the whole module rather than failing collection on hosts where
# any of those are missing.
PyQt6 = pytest.importorskip("PyQt6")
matplotlib = pytest.importorskip("matplotlib")
pytest.importorskip("matplotlib.pyplot")
pytest.importorskip("numpy")

from PyQt6 import QtWidgets  # noqa: E402

from src.shared.python.launcher_embed import (  # noqa: E402
    EmbedCapabilities,
    EmbeddableTool,
)


_APPS_PKG_NAME = "engines.Simscape_Multibody_Models.3D_Golf_Model.python.src.apps"
_ADAPTER_MOD_NAME = f"{_APPS_PKG_NAME}._embed_adapter"


def _import_apps_pkg():
    """Import the engine ``apps`` package and return it.

    Uses the same dotted name that ``test_c3d_viewer_headless.py`` uses
    so the package's relative imports (``from ...c3d_reader``) resolve
    to the correct ancestor without needing the
    ``run_c3d_viewer.py`` ``sys.path`` pivots. ``importlib.import_module``
    accepts dotted segments like ``3D_Golf_Model`` even though they are
    not valid Python identifiers.
    """
    try:
        return importlib.import_module(_APPS_PKG_NAME)
    except ImportError as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"apps package not importable: {exc}")


@pytest.fixture
def adapter():
    """Construct a fresh adapter instance for each test.

    Importing the adapter module does *not* trigger registration — only
    importing the parent ``apps`` package does. Tests that need to assert
    on the registry side-effect import the parent package explicitly.
    """
    pkg = _import_apps_pkg()
    adapter_mod = importlib.import_module(_ADAPTER_MOD_NAME)
    yield adapter_mod._C3DViewerEmbedAdapter()
    # Belt-and-brace: drop matplotlib figures so test ordering does not
    # leave globals dirty for the next test.
    import matplotlib.pyplot as plt

    plt.close("all")
    del pkg


@pytest.mark.unit
def test_adapter_satisfies_embeddable_tool_protocol(adapter) -> None:
    """The adapter is a structural :class:`EmbeddableTool`."""
    assert isinstance(adapter, EmbeddableTool)
    assert adapter.tool_id == "c3d_viewer"


@pytest.mark.unit
def test_embed_capabilities_match_models_yaml(adapter) -> None:
    caps = adapter.embed_capabilities()
    assert isinstance(caps, EmbedCapabilities)
    assert caps.supports_embedded is True
    assert caps.prefers_dock is False
    assert caps.min_size == (900, 600)
    assert caps.requires_separate_qapplication is False


@pytest.mark.unit
def test_is_dirty_is_false(adapter) -> None:
    """The viewer is read-only; ``is_dirty`` always returns False."""
    assert adapter.is_dirty() is False


@pytest.mark.unit
def test_create_main_widget_returns_qwidget(qapp, adapter) -> None:
    """``create_main_widget(None)`` returns a ``QWidget``."""
    widget = adapter.create_main_widget(None)
    try:
        assert isinstance(widget, QtWidgets.QWidget)
    finally:
        widget.deleteLater()


@pytest.mark.unit
def test_cleanup_closes_all_matplotlib_figures(adapter) -> None:
    """``cleanup`` releases every open matplotlib figure."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.close("all")
    plt.figure()
    plt.figure()
    plt.figure()
    assert len(plt.get_fignums()) == 3

    adapter.cleanup()

    assert len(plt.get_fignums()) == 0


@pytest.mark.unit
def test_cleanup_is_idempotent(adapter) -> None:
    """Calling ``cleanup`` twice does not raise."""
    adapter.cleanup()
    adapter.cleanup()  # no-op, must not raise


@pytest.mark.unit
def test_apps_package_import_registers_adapter() -> None:
    """Importing the ``apps`` package registers the adapter."""
    from src.shared.python.launcher_embed import (
        EMBEDDABLE_TOOL_REGISTRY,
        get_embeddable_tool,
        unregister_embeddable_tool,
    )

    # Clear any prior registration so we observe the import-time effect
    # cleanly. Other tools in the registry are left untouched.
    if "c3d_viewer" in EMBEDDABLE_TOOL_REGISTRY:
        unregister_embeddable_tool("c3d_viewer")

    # Re-import the package; ``apps/__init__.py`` registers the adapter.
    sys.modules.pop(_APPS_PKG_NAME, None)
    sys.modules.pop(_ADAPTER_MOD_NAME, None)
    _import_apps_pkg()

    registered = get_embeddable_tool("c3d_viewer")
    assert registered is not None
    assert registered.tool_id == "c3d_viewer"
    assert isinstance(registered, EmbeddableTool)
