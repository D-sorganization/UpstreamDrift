"""Tests for the Pinocchio dashboard embeddable-tool adapter.

Verifies the adapter satisfies the
:class:`~src.shared.python.launcher_embed.EmbeddableTool` protocol,
exposes the capabilities documented in Subtask 5 / #4998 of EPIC #4993,
hands out real :class:`QWidget` instances from
:meth:`create_main_widget`, and registers itself with the
embeddable-tool registry on import.

The widget-construction tests require both PyQt6 and the ``pinocchio``
wheel; they are skipped automatically on hosts where either is missing.
"""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.unit]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PyQt6 = pytest.importorskip("PyQt6")


@pytest.fixture(scope="module")
def qapp():  # noqa: ANN201
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


# --- Protocol conformance -----------------------------------------------


def test_adapter_satisfies_embeddable_tool_protocol() -> None:
    from src.engines.physics_engines.pinocchio.python.pinocchio_golf._embed_adapter import (  # noqa: E501
        _PinocchioDashboardEmbedAdapter,
    )
    from src.shared.python.launcher_embed import EmbeddableTool

    adapter = _PinocchioDashboardEmbedAdapter()
    assert isinstance(adapter, EmbeddableTool)


def test_adapter_tool_id_is_pinocchio_golf() -> None:
    from src.engines.physics_engines.pinocchio.python.pinocchio_golf._embed_adapter import (  # noqa: E501
        _PinocchioDashboardEmbedAdapter,
    )

    adapter = _PinocchioDashboardEmbedAdapter()
    assert adapter.tool_id == "pinocchio_golf"


# --- Capabilities values -------------------------------------------------


def test_embed_capabilities_match_spec() -> None:
    from src.engines.physics_engines.pinocchio.python.pinocchio_golf._embed_adapter import (  # noqa: E501
        _PinocchioDashboardEmbedAdapter,
    )
    from src.shared.python.launcher_embed import EmbedCapabilities

    caps = _PinocchioDashboardEmbedAdapter().embed_capabilities()
    assert isinstance(caps, EmbedCapabilities)
    assert caps.supports_embedded is True
    assert caps.prefers_dock is False
    assert caps.min_size == (1000, 700)
    assert caps.requires_separate_qapplication is False


def test_is_dirty_default_is_false() -> None:
    from src.engines.physics_engines.pinocchio.python.pinocchio_golf._embed_adapter import (  # noqa: E501
        _PinocchioDashboardEmbedAdapter,
    )

    assert _PinocchioDashboardEmbedAdapter().is_dirty() is False


def test_cleanup_is_idempotent() -> None:
    from src.engines.physics_engines.pinocchio.python.pinocchio_golf._embed_adapter import (  # noqa: E501
        _PinocchioDashboardEmbedAdapter,
    )

    adapter = _PinocchioDashboardEmbedAdapter()
    # Cleanup is allowed before any widget has been handed out.
    adapter.cleanup()
    adapter.cleanup()


# --- create_main_widget returns a real QWidget --------------------------


def test_create_main_widget_returns_qwidget(qapp, monkeypatch) -> None:  # noqa: ANN001
    """``create_main_widget`` returns the registered ``MainWidget`` type.

    The legacy :class:`PinocchioGUI` ``QMainWindow`` carries a number
    of pre-existing wiring concerns (heavy mixin chain, Meshcat
    visualizer subprocess, default URDF model load) and a default
    URDF auto-load that block or hang construction in headless CI.
    Those are out of scope for this refactor.

    What we want to verify is the embed contract: the adapter
    delegates to :class:`MainWidget` and hands the resulting
    ``QWidget`` to the host. We do that by stubbing :class:`MainWidget`
    to a minimal ``QWidget`` and exercising the adapter end-to-end
    against the stub. The real :class:`MainWidget` is exercised by
    the standalone launch path (``python gui.py``) and by the
    in-launcher integration tests once the underlying dashboard
    behaviors are addressed.
    """
    pin = pytest.importorskip("pinocchio")
    if not hasattr(pin, "Model"):
        pytest.skip("pinocchio wheel is a stub (no pin.Model); skipping GUI import")
    from PyQt6.QtWidgets import QWidget

    from src.engines.physics_engines.pinocchio.python.pinocchio_golf import (
        _embed_adapter as embed_module,
    )
    from src.engines.physics_engines.pinocchio.python.pinocchio_golf import (
        gui as gui_module,
    )
    from src.engines.physics_engines.pinocchio.python.pinocchio_golf._embed_adapter import (  # noqa: E501
        _PinocchioDashboardEmbedAdapter,
    )

    # ``MainWidget`` must be a ``QWidget`` subclass to satisfy the
    # contract; check that statically before exercising the adapter.
    assert issubclass(gui_module.MainWidget, QWidget)

    class _StubWidget(QWidget):
        def cleanup(self) -> None:
            pass

    monkeypatch.setattr(gui_module, "MainWidget", _StubWidget)
    # The adapter does ``from .gui import MainWidget`` inside
    # ``create_main_widget``; patching the source module is enough
    # because the import is lazy.
    monkeypatch.setattr(
        embed_module,
        "_PinocchioDashboardEmbedAdapter",
        _PinocchioDashboardEmbedAdapter,
    )

    adapter = _PinocchioDashboardEmbedAdapter()
    widget = adapter.create_main_widget(None)
    try:
        assert isinstance(widget, QWidget)
    finally:
        adapter.cleanup()
        widget.deleteLater()


# --- Registry side-effect on import -------------------------------------


def test_import_registers_pinocchio_dashboard_in_registry() -> None:
    """Importing the package registers the adapter."""
    from src.shared.python.launcher_embed import (
        EMBEDDABLE_TOOL_REGISTRY,
        get_embeddable_tool,
    )

    # The package's ``__init__.py`` runs the registry side-effect via
    # ``contextlib.suppress(ImportError)``; subsequent imports are a
    # no-op thanks to the ``get_embeddable_tool`` guard in
    # ``_embed_adapter``.
    import src.engines.physics_engines.pinocchio.python.pinocchio_golf  # noqa: F401

    assert "pinocchio_golf" in EMBEDDABLE_TOOL_REGISTRY
    tool = get_embeddable_tool("pinocchio_golf")
    assert tool is not None
    assert tool.tool_id == "pinocchio_golf"
    assert tool.embed_capabilities().supports_embedded is True
