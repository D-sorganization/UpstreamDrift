"""Tests for run_launcher event-loop and QApplication ownership (#9165).

Verifies:
- When QApplication is pre-existing (e.g. in tests/embedded hosts), run_launcher does not call app.exec() and returns 0.
- When QApplication is created by run_launcher, app.exec() is called and its exit code returned.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from src.launchers.base import run_launcher


@pytest.mark.unit
def test_run_launcher_reuses_existing_qapp_without_exec() -> None:
    """When QApplication exists, run_launcher does not enter an inner event loop."""
    mock_app = MagicMock()
    mock_launcher_cls = MagicMock()

    mock_qapp_cls = MagicMock()
    mock_qapp_cls.instance.return_value = mock_app

    with patch("src.launchers.base.QApplication", mock_qapp_cls):
        exit_code = run_launcher(mock_launcher_cls)
        assert exit_code == 0
        mock_app.exec.assert_not_called()
        mock_launcher_cls.assert_called_once()
        mock_launcher_cls.return_value.init_ui.assert_called_once()
        mock_launcher_cls.return_value.show.assert_called_once()


@pytest.mark.unit
def test_run_launcher_executes_when_it_created_app() -> None:
    """When run_launcher creates the QApplication, it calls app.exec() and returns the code."""
    mock_app = MagicMock()
    mock_app.exec.return_value = 42
    mock_launcher_cls = MagicMock()

    mock_qapp_cls = MagicMock()
    mock_qapp_cls.instance.return_value = None
    mock_qapp_cls.return_value = mock_app

    with patch("src.launchers.base.QApplication", mock_qapp_cls):
        exit_code = run_launcher(mock_launcher_cls)
        assert exit_code == 42
        mock_qapp_cls.assert_called_once()
        mock_app.setStyle.assert_called_once_with("Fusion")
        mock_app.exec.assert_called_once()
        mock_launcher_cls.assert_called_once()
        mock_launcher_cls.return_value.init_ui.assert_called_once()
        mock_launcher_cls.return_value.show.assert_called_once()
