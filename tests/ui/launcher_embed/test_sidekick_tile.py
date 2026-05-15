"""Headless smoke test for the Sidekick embeddable-tool adapter.

Exercised under ``QT_QPA_PLATFORM=offscreen``; the test skips cleanly
when PyQt6 (or the AI session-manager dependency stack) is not
available. Mirrors the pattern in ``tests/ui/launcher_embed``: a
minimal ``QApplication`` is constructed in the session-scoped ``qapp``
fixture, then ``create_main_widget(None)`` is called and the returned
:class:`QWidget` is verified.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PyQt6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = [pytest.mark.unit]


def test_create_main_widget_returns_qwidget(qapp) -> None:  # noqa: ANN001
    """``create_main_widget`` builds a real :class:`QWidget`."""
    pytest.importorskip("src.shared.python.ai.gui.assistant_panel")
    from PyQt6.QtWidgets import QWidget  # noqa: E402

    from src.tools.sidekick._embed_adapter import _SidekickEmbedAdapter

    adapter = _SidekickEmbedAdapter()
    widget = adapter.create_main_widget(None)
    try:
        assert isinstance(widget, QWidget)
    finally:
        adapter.cleanup()


def test_create_main_widget_accepts_parent(qapp) -> None:  # noqa: ANN001
    """The created widget is parented to the supplied parent."""
    pytest.importorskip("src.shared.python.ai.gui.assistant_panel")
    from PyQt6.QtWidgets import QWidget  # noqa: E402

    from src.tools.sidekick._embed_adapter import _SidekickEmbedAdapter

    parent = QWidget()
    adapter = _SidekickEmbedAdapter()
    try:
        widget = adapter.create_main_widget(parent)
        assert widget.parent() is parent
    finally:
        adapter.cleanup()
        parent.deleteLater()
