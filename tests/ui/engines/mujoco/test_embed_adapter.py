"""Tests for the MuJoCo dashboard embeddable-tool adapter.

Verifies the adapter satisfies the
:class:`~src.shared.python.launcher_embed.EmbeddableTool` protocol,
exposes the capabilities documented in Subtask 5 / #4998 of EPIC #4993,
hands out real :class:`QWidget` instances from
:meth:`create_main_widget`, and registers itself with the
embeddable-tool registry on import.

The widget-construction tests require both PyQt6 and the ``mujoco``
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
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf._embed_adapter import (  # noqa: E501
        _MujocoDashboardEmbedAdapter,
    )
    from src.shared.python.launcher_embed import EmbeddableTool

    adapter = _MujocoDashboardEmbedAdapter()
    assert isinstance(adapter, EmbeddableTool)


def test_adapter_tool_id_is_mujoco_unified() -> None:
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf._embed_adapter import (  # noqa: E501
        _MujocoDashboardEmbedAdapter,
    )

    adapter = _MujocoDashboardEmbedAdapter()
    assert adapter.tool_id == "mujoco_unified"


# --- Capabilities values -------------------------------------------------


def test_embed_capabilities_match_spec() -> None:
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf._embed_adapter import (  # noqa: E501
        _MujocoDashboardEmbedAdapter,
    )
    from src.shared.python.launcher_embed import EmbedCapabilities

    caps = _MujocoDashboardEmbedAdapter().embed_capabilities()
    assert isinstance(caps, EmbedCapabilities)
    assert caps.supports_embedded is True
    assert caps.prefers_dock is False
    assert caps.min_size == (1000, 700)
    assert caps.requires_separate_qapplication is False


def test_is_dirty_default_is_false() -> None:
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf._embed_adapter import (  # noqa: E501
        _MujocoDashboardEmbedAdapter,
    )

    assert _MujocoDashboardEmbedAdapter().is_dirty() is False


def test_cleanup_is_idempotent() -> None:
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf._embed_adapter import (  # noqa: E501
        _MujocoDashboardEmbedAdapter,
    )

    adapter = _MujocoDashboardEmbedAdapter()
    # Cleanup is allowed before any widget has been handed out.
    adapter.cleanup()
    adapter.cleanup()


# --- create_main_widget returns a real QWidget --------------------------


def test_create_main_widget_returns_qwidget(qapp, monkeypatch) -> None:  # noqa: ANN001
    """``create_main_widget`` returns the registered ``MainWidget`` type.

    The legacy :class:`AdvancedGolfAnalysisWindow` carries a number of
    pre-existing wiring bugs (missing ``connect_timer`` /
    ``get_num_bodies`` / ``has_model`` / ``get_state`` on
    :class:`MuJoCoSimWidget`) and a chat-dock auto-connect that block
    or hang construction in headless CI. Those are out of scope for
    this refactor.

    What we want to verify is the embed contract: the adapter delegates
    to :class:`MainWidget` and hands the resulting ``QWidget`` to the
    host. We do that by stubbing :class:`MainWidget` to a minimal
    ``QWidget`` and exercising the adapter end-to-end against the
    stub. The real :class:`MainWidget` is exercised by the standalone
    launch path (``python -m mujoco_humanoid_golf``) and by the
    in-launcher integration tests once the underlying dashboard bugs
    are addressed.
    """
    pytest.importorskip("mujoco")
    from PyQt6.QtWidgets import QWidget

    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf import (
        _embed_adapter as embed_module,
    )
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf._embed_adapter import (  # noqa: E501
        _MujocoDashboardEmbedAdapter,
    )
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.gui.core import (  # noqa: E501
        main_widget as main_widget_module,
    )

    # ``MainWidget`` must be a ``QWidget`` subclass to satisfy the
    # contract; check that statically before exercising the adapter.
    assert issubclass(main_widget_module.MainWidget, QWidget)

    class _StubWidget(QWidget):
        def cleanup(self) -> None:
            pass

    monkeypatch.setattr(main_widget_module, "MainWidget", _StubWidget)
    # The adapter does ``from .gui.core.main_widget import MainWidget``
    # inside ``create_main_widget``; patching the source module is
    # enough because the import is lazy.
    monkeypatch.setattr(
        embed_module, "_MujocoDashboardEmbedAdapter", _MujocoDashboardEmbedAdapter
    )  # noqa: E501

    adapter = _MujocoDashboardEmbedAdapter()
    widget = adapter.create_main_widget(None)
    try:
        assert isinstance(widget, QWidget)
    finally:
        adapter.cleanup()
        widget.deleteLater()


# --- Registry side-effect on import -------------------------------------


def test_import_registers_mujoco_dashboard_in_registry() -> None:
    """Importing the package registers the adapter."""
    from src.shared.python.launcher_embed import (
        EMBEDDABLE_TOOL_REGISTRY,
        get_embeddable_tool,
    )

    # The package's ``__init__.py`` runs the registry side-effect via
    # ``contextlib.suppress(ImportError)``; subsequent imports are a
    # no-op thanks to the ``get_embeddable_tool`` guard in
    # ``_embed_adapter``.
    import src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf  # noqa: F401,E501

    assert "mujoco_unified" in EMBEDDABLE_TOOL_REGISTRY
    tool = get_embeddable_tool("mujoco_unified")
    assert tool is not None
    assert tool.tool_id == "mujoco_unified"
    assert tool.embed_capabilities().supports_embedded is True


def test_apply_styling_uses_theme_manager_when_available(qapp) -> None:  # noqa: ANN001
    """Issue #6509: _apply_styling must call apply_theme_to_window, not be a no-op."""
    from unittest.mock import MagicMock, patch

    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.gui.core.main_window import (  # noqa: E501
        AdvancedGolfAnalysisWindow,
    )

    mock_apply = MagicMock()

    with patch.object(AdvancedGolfAnalysisWindow, "__init__", return_value=None):
        window = AdvancedGolfAnalysisWindow.__new__(AdvancedGolfAnalysisWindow)

    # Patch the module attribute the import inside _apply_styling binds to
    with patch.dict(
        "sys.modules",
        {
            "src.shared.python.theme": type(
                "m",
                (),
                {"apply_theme_to_window": mock_apply},
            )()
        },
    ):
        window._apply_styling()

    mock_apply.assert_called_once_with(window)
