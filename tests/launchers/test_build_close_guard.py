"""Tests for shared active-build close handling."""

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("PyQt6")
pytestmark = pytest.mark.unit

from PyQt6.QtWidgets import QMessageBox, QWidget

from src.launchers.build_close_guard import confirm_cancel_running_build_for_close


def test_close_guard_ignores_event_when_user_keeps_build_running(qapp) -> None:
    parent = QWidget()
    event = MagicMock()
    build_thread = MagicMock()
    build_thread.isRunning.return_value = True

    with patch(
        "src.launchers.build_close_guard.QMessageBox.question",
        return_value=QMessageBox.StandardButton.No,
    ):
        should_close = confirm_cancel_running_build_for_close(
            parent,
            event,
            build_thread,
            log_message="cancel build for test",
        )

    assert should_close is False
    event.ignore.assert_called_once()
    build_thread.cancel.assert_not_called()
    build_thread.wait.assert_not_called()


def test_close_guard_cancels_and_waits_when_user_confirms(qapp) -> None:
    parent = QWidget()
    event = MagicMock()
    build_thread = MagicMock()
    build_thread.isRunning.return_value = True

    with patch(
        "src.launchers.build_close_guard.QMessageBox.question",
        return_value=QMessageBox.StandardButton.Yes,
    ):
        should_close = confirm_cancel_running_build_for_close(
            parent,
            event,
            build_thread,
            log_message="cancel build for test",
        )

    assert should_close is True
    event.ignore.assert_not_called()
    build_thread.cancel.assert_called_once()
    build_thread.wait.assert_called_once()


def test_close_guard_allows_close_without_active_thread(qapp) -> None:
    parent = QWidget()
    event = MagicMock()
    build_thread = MagicMock()
    build_thread.isRunning.return_value = False

    assert (
        confirm_cancel_running_build_for_close(
            parent,
            event,
            build_thread,
            log_message="cancel build for test",
        )
        is True
    )
    build_thread.cancel.assert_not_called()
