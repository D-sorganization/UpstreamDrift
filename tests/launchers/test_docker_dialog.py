"""Tests for docker_dialog.py."""

from unittest.mock import MagicMock, patch

import pytest

# Ensure PyQt classes are available
pytest.importorskip("PyQt6")

from src.launchers.docker_dialog import EnvironmentDialog


@pytest.fixture
def dialog(qapp):
    """Provide an EnvironmentDialog instance."""
    return EnvironmentDialog()


def test_init(dialog):
    """Test dialog initialization."""
    assert dialog.windowTitle() == "Manage Environment"
    assert dialog.build_thread is None
    assert dialog._build_start_time == 0.0
    assert dialog._elapsed_timer_id is None


def test_setup_ui(dialog):
    """Test UI setup."""
    assert dialog.combo_stage is not None
    assert dialog.btn_build is not None
    assert dialog.btn_cancel is not None
    assert dialog.build_status_label is not None
    assert dialog.console is not None
    assert dialog.btn_cancel.isEnabled() is False


@patch("src.launchers.docker_dialog.DockerBuildThread")
def test_start_build(mock_thread_cls, dialog):
    """Test starting the build process."""
    mock_thread = MagicMock()
    mock_thread_cls.return_value = mock_thread

    # Make startTimer a mock or track it
    with patch.object(dialog, "startTimer", return_value=123) as mock_timer:
        dialog.start_build()

        assert dialog.btn_build.isEnabled() is False
        assert dialog.btn_cancel.isEnabled() is True
        assert dialog._elapsed_timer_id == 123
        mock_timer.assert_called_once_with(1000)

        # Ensure the thread was created and started
        mock_thread_cls.assert_called_once()
        mock_thread.start.assert_called_once()
        assert dialog.build_thread == mock_thread


def test_on_build_log(dialog):
    """Test appending to log."""
    # Append creates text without changing read-only
    with patch.object(dialog.console, "append") as mock_append:
        dialog._on_build_log("test log")
        mock_append.assert_called_once_with("test log")

    # Test when verticalScrollBar is None
    with patch.object(dialog.console, "verticalScrollBar", return_value=None):
        dialog._on_build_log("test log 2")


def test_on_build_finished_success(dialog):
    """Test handling a successful build finish."""
    dialog._build_start_time = 1.0
    dialog._elapsed_timer_id = 999

    with (
        patch("src.launchers.docker_dialog.time.monotonic", return_value=3.0),
        patch.object(dialog, "killTimer") as mock_kill,
    ):
        dialog._on_build_finished(True, "Done!")

        assert dialog.btn_build.isEnabled() is True
        assert dialog.btn_cancel.isEnabled() is False
        mock_kill.assert_called_once_with(999)
        assert dialog._elapsed_timer_id is None
        assert "SUCCESS" in dialog.build_status_label.text()
        assert "Done!" in dialog.build_status_label.text()


def test_on_build_finished_failure(dialog):
    """Test handling a failed build finish."""
    dialog._build_start_time = 0.0
    dialog._elapsed_timer_id = None

    with patch.object(dialog, "killTimer") as mock_kill:
        dialog._on_build_finished(False, "Error!")

        assert dialog.btn_build.isEnabled() is True
        assert dialog.btn_cancel.isEnabled() is False
        mock_kill.assert_not_called()
        assert "FAILED" in dialog.build_status_label.text()
        assert "Error!" in dialog.build_status_label.text()


def test_cancel_build(dialog):
    """Test canceling an active build."""
    mock_thread = MagicMock()
    mock_thread.isRunning.return_value = True
    dialog.build_thread = mock_thread
    dialog._elapsed_timer_id = 111

    with patch.object(dialog, "killTimer") as mock_kill:
        dialog._cancel_build()

        mock_thread.terminate.assert_called_once()
        assert dialog.btn_build.isEnabled() is True
        assert dialog.btn_cancel.isEnabled() is False
        mock_kill.assert_called_once_with(111)
        assert dialog._elapsed_timer_id is None
        assert dialog.build_status_label.text() == "Build cancelled."

    # Test canceling when _elapsed_timer_id is None
    mock_thread.reset_mock()
    dialog._cancel_build()
    mock_thread.terminate.assert_called_once()
    assert dialog._elapsed_timer_id is None


def test_cancel_build_no_thread(dialog):
    """Test canceling when no build thread is active."""
    dialog.build_thread = None
    dialog._cancel_build()
    # Nothing should crash


def test_timer_event(dialog):
    """Test timer event updates elapsed time."""
    dialog._build_start_time = 5.0
    with patch("src.launchers.docker_dialog.time.monotonic", return_value=15.0):
        dialog.timerEvent(MagicMock())
        assert "10s elapsed" in dialog.build_status_label.text()
