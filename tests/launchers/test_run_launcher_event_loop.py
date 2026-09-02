"""Tests for run_launcher event-loop and QApplication ownership (#9165).

Verifies:
- When QApplication is pre-existing (e.g. in tests/embedded hosts), run_launcher does not call app.exec() and returns 0.
- When QApplication is created by run_launcher, app.exec() is called and its exit code returned.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from PyQt6.QtWidgets import QApplication
from src.launchers.base import BaseLauncher, run_launcher


class DummyLauncher(BaseLauncher):
    """Minimal concrete BaseLauncher for testing."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.init_ui_called = False

    def init_ui(self) -> None:
        self.init_ui_called = True


@pytest.mark.unit
def test_run_launcher_reuses_existing_qapp_without_exec(qapp: QApplication) -> None:
    """When QApplication exists, run_launcher does not enter an inner event loop."""
    with patch.object(qapp, "exec", return_value=123) as mock_exec:
        exit_code = run_launcher(DummyLauncher)
        assert exit_code == 0
        mock_exec.assert_not_called()


@pytest.mark.unit
def test_run_launcher_executes_when_it_created_app() -> None:
    """When run_launcher creates the QApplication, it calls app.exec() and returns the code."""
    mock_app = MagicMock()
    mock_app.exec.return_value = 42

    with (
        patch("src.launchers.base.QApplication") as mock_qapp_class,
        patch.object(DummyLauncher, "center_window"),
        patch.object(DummyLauncher, "show"),
    ):
        mock_qapp_class.instance.return_value = None
        mock_qapp_class.return_value = mock_app
        exit_code = run_launcher(DummyLauncher)
        assert exit_code == 42
        mock_qapp_class.assert_called_once()
        mock_app.exec.assert_called_once()
