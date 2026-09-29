"""Tests for the Motion-Match Preview embeddable-tool adapter.

Verifies the adapter satisfies the
:class:`~src.shared.python.launcher_embed.EmbeddableTool` protocol,
exposes the capabilities documented in Subtask 5 / #4998, hands out
real :class:`QWidget` instances from :meth:`create_main_widget`, and
registers itself with the embeddable-tool registry on import.

The package is named ``starting_pose_matcher`` for historical reasons
but its launcher tile id is ``motion_target_preview`` — both the
adapter and the registry use the latter.
"""

from __future__ import annotations

import os

import pytest

from src.shared.python.engine_core.engine_availability import (
    skip_if_unavailable,
)

pytestmark = [
    skip_if_unavailable("pyqt6"),
    pytest.mark.unit,
]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():  # noqa: ANN201
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


# --- Protocol conformance -----------------------------------------------


def test_adapter_satisfies_embeddable_tool_protocol() -> None:
    from src.shared.python.launcher_embed import EmbeddableTool
    from src.tools.starting_pose_matcher._embed_adapter import (
        _MotionMatchPreviewEmbedAdapter,
    )

    adapter = _MotionMatchPreviewEmbedAdapter()
    assert isinstance(adapter, EmbeddableTool)


def test_adapter_tool_id_is_motion_target_preview() -> None:
    from src.tools.starting_pose_matcher._embed_adapter import (
        _MotionMatchPreviewEmbedAdapter,
    )

    adapter = _MotionMatchPreviewEmbedAdapter()
    assert adapter.tool_id == "motion_target_preview"


# --- Capabilities values -------------------------------------------------


def test_embed_capabilities_match_spec() -> None:
    from src.shared.python.launcher_embed import EmbedCapabilities
    from src.tools.starting_pose_matcher._embed_adapter import (
        _MotionMatchPreviewEmbedAdapter,
    )

    caps = _MotionMatchPreviewEmbedAdapter().embed_capabilities()
    assert isinstance(caps, EmbedCapabilities)
    assert caps.supports_embedded is True
    assert caps.prefers_dock is False
    # Large GUI — reserves room for the 3D viewport plus the per-section
    # control column.
    assert caps.min_size == (1024, 720)
    assert caps.requires_separate_qapplication is False


def test_is_dirty_default_is_false() -> None:
    from src.tools.starting_pose_matcher._embed_adapter import (
        _MotionMatchPreviewEmbedAdapter,
    )

    assert _MotionMatchPreviewEmbedAdapter().is_dirty() is False


def test_cleanup_is_idempotent() -> None:
    from src.tools.starting_pose_matcher._embed_adapter import (
        _MotionMatchPreviewEmbedAdapter,
    )

    adapter = _MotionMatchPreviewEmbedAdapter()
    # Cleanup is allowed before any widget has been handed out.
    adapter.cleanup()
    adapter.cleanup()


# --- create_main_widget returns a real QWidget --------------------------


def test_create_main_widget_returns_qwidget(qapp) -> None:  # noqa: ANN001
    from PyQt6.QtWidgets import QWidget

    from src.tools.starting_pose_matcher._embed_adapter import (
        _MotionMatchPreviewEmbedAdapter,
    )

    adapter = _MotionMatchPreviewEmbedAdapter()
    widget = adapter.create_main_widget(None)
    try:
        assert isinstance(widget, QWidget)
    finally:
        adapter.cleanup()
        widget.deleteLater()


def test_create_main_widget_accepts_parent(qapp) -> None:  # noqa: ANN001
    from PyQt6.QtWidgets import QWidget

    from src.tools.starting_pose_matcher._embed_adapter import (
        _MotionMatchPreviewEmbedAdapter,
    )

    parent = QWidget()
    adapter = _MotionMatchPreviewEmbedAdapter()
    try:
        widget = adapter.create_main_widget(parent)
        assert widget.parent() is parent
    finally:
        adapter.cleanup()
        parent.deleteLater()


# --- Registry side-effect on import -------------------------------------


def test_import_registers_motion_target_preview_in_registry() -> None:
    """Importing :mod:`src.tools.starting_pose_matcher` registers the adapter."""
    from src.shared.python.launcher_embed import (
        EMBEDDABLE_TOOL_REGISTRY,
        get_embeddable_tool,
    )

    # The package's __init__.py side-effect runs at first import; subsequent
    # imports are a no-op thanks to the ``get_embeddable_tool`` guard.
    import src.tools.starting_pose_matcher  # noqa: F401

    assert "motion_target_preview" in EMBEDDABLE_TOOL_REGISTRY
    tool = get_embeddable_tool("motion_target_preview")
    assert tool is not None
    assert tool.tool_id == "motion_target_preview"
    assert tool.embed_capabilities().supports_embedded is True
