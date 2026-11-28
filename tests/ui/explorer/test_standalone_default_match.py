"""Offscreen: the standalone Model Explorer opens the default matched swing."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6", reason="the explorer needs PyQt6")

from PyQt6.QtWidgets import QApplication  # noqa: E402

pytestmark = [pytest.mark.ui, pytest.mark.integration]


@pytest.fixture
def qapp() -> QApplication:
    inst = QApplication.instance()
    return inst if isinstance(inst, QApplication) else QApplication(["standalone"])


def test_standalone_window_queues_the_default_match_after_it_shows(
    qapp: QApplication,
) -> None:
    """Like the launcher tab: paint first, then load the best match (#15150)."""
    from src.tools.model_explorer import main_window
    from src.tools.model_explorer.gui import MainWidget

    with patch.object(MainWidget, "load_default_match", return_value=True) as load:
        window = main_window.create_standalone_window()
        assert window.isVisible()
        load.assert_not_called()  # queued, so the window paints first
        QApplication.processEvents()
        load.assert_called_once_with()
    window.close()
    window.deleteLater()
