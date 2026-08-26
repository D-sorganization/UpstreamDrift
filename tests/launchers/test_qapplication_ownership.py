"""Regression contracts for process-wide QApplication ownership."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from src.launchers.base import run_launcher


@pytest.mark.unit
def test_run_launcher_reuses_process_application_without_entering_event_loop() -> None:
    """An embedded launcher must not construct or execute a second application."""
    application_class = MagicMock()
    existing_application = MagicMock()
    application_class.instance.return_value = existing_application
    launcher_class = MagicMock()

    with patch.dict(run_launcher.__globals__, {"QApplication": application_class}):
        result = run_launcher(launcher_class)

    assert result == 0
    application_class.instance.assert_called_once_with()
    application_class.assert_not_called()
    existing_application.setStyle.assert_called_once_with("Fusion")
    existing_application.exec.assert_not_called()
    launcher_class.assert_called_once_with()


@pytest.mark.unit
def test_run_launcher_executes_event_loop_for_application_it_creates() -> None:
    """A standalone launcher must execute the application event loop it owns."""
    application_class = MagicMock()
    application_class.instance.return_value = None
    owned_application = application_class.return_value
    owned_application.exec.return_value = 17
    launcher_class = MagicMock()

    with patch.dict(run_launcher.__globals__, {"QApplication": application_class}):
        result = run_launcher(launcher_class)

    assert result == 17
    application_class.instance.assert_called_once_with()
    application_class.assert_called_once_with(sys.argv)
    owned_application.setStyle.assert_called_once_with("Fusion")
    owned_application.exec.assert_called_once_with()
    launcher_class.assert_called_once_with()
