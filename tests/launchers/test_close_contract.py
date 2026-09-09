"""Regression tests for #8895 / #8896 — one close contract, no lost work.

Three defects with one shape: a close affordance that bypasses the guard
that was supposed to run on it.

* #8895 `EnvironmentDialog`'s "Close" button was wired to `self.accept`.
  `QDialog.accept()` calls `done()`, which hides the dialog **without**
  dispatching a `QCloseEvent`, so `closeEvent` -- and its running-build
  guard -- never ran, while the window-manager X *did* run it. One dialog,
  two closes, opposite semantics: the button orphaned a running
  `docker build`.
* #8895 the guard itself then called `build_thread.wait()` with no timeout
  on the GUI thread, after a `cancel()` that already blocks for up to ~4 s.
* #8896 `SettingsDialog`'s footer Close was wired to `self.reject` -- the
  same `done()` path -- and there was no dirty check at all, so unsaved
  preference edits vanished silently. It also meant the widget's own build
  guard was unreachable from the dialog, because a QDialog does not deliver
  close events to its children.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("PyQt6")
pytestmark = pytest.mark.unit

from PyQt6.QtWidgets import QMessageBox  # noqa: E402

from src.launchers.build_close_guard import (  # noqa: E402
    BUILD_CANCEL_JOIN_TIMEOUT_MS,
    confirm_cancel_running_build_for_close,
)


# ----------------------------------------------------------------------
# #8895 — the guard bounds its join and never blocks unbounded
# ----------------------------------------------------------------------


def test_guard_joins_with_a_timeout_not_forever(qapp) -> None:
    """`wait()` used to be untimed on the GUI thread."""
    from PyQt6.QtWidgets import QWidget

    parent = QWidget()
    event = MagicMock()
    build_thread = MagicMock()
    build_thread.isRunning.return_value = True
    build_thread.wait.return_value = True

    with patch(
        "src.launchers.build_close_guard.QMessageBox.question",
        return_value=QMessageBox.StandardButton.Yes,
    ):
        assert (
            confirm_cancel_running_build_for_close(
                parent, event, build_thread, log_message="cancel for test"
            )
            is True
        )

    build_thread.cancel.assert_called_once()
    build_thread.wait.assert_called_once_with(BUILD_CANCEL_JOIN_TIMEOUT_MS)
    assert BUILD_CANCEL_JOIN_TIMEOUT_MS > 0


def test_guard_tells_the_user_when_the_thread_outlives_the_join(qapp) -> None:
    """A thread that will not stop must not silently freeze the window."""
    from PyQt6.QtWidgets import QWidget

    parent = QWidget()
    event = MagicMock()
    build_thread = MagicMock()
    build_thread.isRunning.return_value = True
    build_thread.wait.return_value = False  # never finished

    with (
        patch(
            "src.launchers.build_close_guard.QMessageBox.question",
            return_value=QMessageBox.StandardButton.Yes,
        ),
        patch("src.launchers.build_close_guard.QMessageBox.information") as information,
    ):
        assert (
            confirm_cancel_running_build_for_close(
                parent, event, build_thread, log_message="cancel for test"
            )
            is True
        )

    assert information.called, "the user must be told the build is still stopping"
    event.ignore.assert_not_called()


def test_guard_restores_the_cursor_even_when_cancel_raises(qapp) -> None:
    """A wait cursor must never outlive a failed cancel."""
    from PyQt6.QtWidgets import QApplication, QWidget

    parent = QWidget()
    event = MagicMock()
    build_thread = MagicMock()
    build_thread.isRunning.return_value = True
    build_thread.cancel.side_effect = RuntimeError("cancel blew up")

    before = QApplication.overrideCursor()
    with (
        patch(
            "src.launchers.build_close_guard.QMessageBox.question",
            return_value=QMessageBox.StandardButton.Yes,
        ),
        pytest.raises(RuntimeError),
    ):
        confirm_cancel_running_build_for_close(
            parent, event, build_thread, log_message="cancel for test"
        )
    assert QApplication.overrideCursor() is before


# ----------------------------------------------------------------------
# #8895 — both close affordances behave identically
# ----------------------------------------------------------------------


def test_docker_dialog_close_button_routes_through_close_event() -> None:
    """The Close button must call `close()`, never `accept()`/`reject()`.

    Asserted on the source rather than by clicking, because the whole
    defect is that `accept()` produces no observable close event to
    assert on -- the dialog just disappears and the build keeps running.
    """
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[2] / "src" / "launchers" / "docker_dialog.py"
    ).read_text(encoding="utf-8")

    assert "close_btn.clicked.connect(self.close)" in source
    assert "close_btn.clicked.connect(self.accept)" not in source


def test_settings_dialog_close_button_routes_through_close_event() -> None:
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[2] / "src" / "launchers" / "settings_dialog.py"
    ).read_text(encoding="utf-8")

    assert "buttons.rejected.connect(self.close)" in source
    assert "buttons.rejected.connect(self.reject)" not in source


def test_docker_dialog_close_triggers_the_guard(qapp) -> None:
    """Clicking Close on a dialog with a running build must prompt."""
    from src.launchers.docker_dialog import EnvironmentDialog

    dialog = EnvironmentDialog()
    try:
        build_thread = MagicMock()
        build_thread.isRunning.return_value = True
        build_thread.wait.return_value = True
        dialog.build_thread = build_thread

        with patch(
            "src.launchers.build_close_guard.QMessageBox.question",
            return_value=QMessageBox.StandardButton.No,
        ) as question:
            dialog.close()

        assert question.called, "Close must reach the running-build guard"
        assert dialog.isVisible() is False or not build_thread.cancel.called
        build_thread.cancel.assert_not_called()
    finally:
        dialog.build_thread = None
        dialog.deleteLater()


# ----------------------------------------------------------------------
# #8896 — the Settings commit model is stated, and Close cannot discard
# ----------------------------------------------------------------------


@pytest.fixture
def settings_widget(qapp):  # noqa: ANN001, ANN201
    from src.launchers.settings_dialog import SettingsWidget

    widget = SettingsWidget()
    yield widget
    widget.deleteLater()


def test_preference_tabs_are_clean_on_open(settings_widget) -> None:  # noqa: ANN001
    assert settings_widget.is_preferences_dirty() is False


def test_editing_a_preference_marks_it_dirty(settings_widget) -> None:  # noqa: ANN001
    """The headline defect: nothing tracked uncommitted preference edits."""
    if not hasattr(settings_widget, "btn_apply"):
        pytest.skip("PreferencesDialog unavailable; deferred tabs were not built")
    settings_widget._mark_preferences_dirty()
    assert settings_widget.is_preferences_dirty() is True
    assert settings_widget.btn_apply.text().endswith("*")
    assert settings_widget.btn_apply.isEnabled()


def test_close_prompts_and_cancel_stops_the_close(settings_widget) -> None:  # noqa: ANN001
    """Close used to discard preference edits with no prompt at all."""
    if not hasattr(settings_widget, "btn_apply"):
        pytest.skip("PreferencesDialog unavailable; deferred tabs were not built")
    settings_widget._mark_preferences_dirty()
    event = MagicMock()

    with patch(
        "src.launchers.settings_close_contract.QMessageBox.question",
        return_value=QMessageBox.StandardButton.Cancel,
    ) as question:
        assert settings_widget.confirm_close(event) is False

    assert question.called
    event.ignore.assert_called_once()
    assert settings_widget.is_preferences_dirty() is True


def test_close_save_answer_applies_the_preferences(settings_widget) -> None:  # noqa: ANN001
    if not hasattr(settings_widget, "btn_apply"):
        pytest.skip("PreferencesDialog unavailable; deferred tabs were not built")
    settings_widget._mark_preferences_dirty()
    event = MagicMock()

    with (
        patch(
            "src.launchers.settings_close_contract.QMessageBox.question",
            return_value=QMessageBox.StandardButton.Save,
        ),
        patch.object(settings_widget._prefs_dialog, "_on_apply") as apply_,
    ):
        assert settings_widget.confirm_close(event) is True

    assert apply_.called, "Save must actually apply before closing"
    assert settings_widget.is_preferences_dirty() is False


def test_close_discard_answer_proceeds(settings_widget) -> None:  # noqa: ANN001
    if not hasattr(settings_widget, "btn_apply"):
        pytest.skip("PreferencesDialog unavailable; deferred tabs were not built")
    settings_widget._mark_preferences_dirty()
    event = MagicMock()

    with patch(
        "src.launchers.settings_close_contract.QMessageBox.question",
        return_value=QMessageBox.StandardButton.Discard,
    ):
        assert settings_widget.confirm_close(event) is True
    event.ignore.assert_not_called()


def test_clean_close_never_prompts(settings_widget) -> None:  # noqa: ANN001
    event = MagicMock()
    with patch(
        "src.launchers.settings_close_contract.QMessageBox.question"
    ) as question:
        assert settings_widget.confirm_close(event) is True
    assert not question.called


def test_commit_model_caption_names_the_current_tab_model(settings_widget) -> None:  # noqa: ANN001
    """The mixed commit model was previously unsignposted anywhere."""
    from src.launchers.settings_close_contract import CAPTION_DEFERRED, CAPTION_LIVE
    from src.launchers.settings_dialog import TAB_APPEARANCE, TAB_LAYOUT

    settings_widget.sync_commit_model_caption(TAB_LAYOUT)
    assert settings_widget.commit_model_label.text() == CAPTION_LIVE

    settings_widget.sync_commit_model_caption(TAB_APPEARANCE)
    assert settings_widget.commit_model_label.text() == CAPTION_DEFERRED


def test_apply_button_is_hidden_on_live_apply_tabs(settings_widget) -> None:  # noqa: ANN001
    """It used to be visible, and meaningless, on every tab."""
    if not hasattr(settings_widget, "btn_apply"):
        pytest.skip("PreferencesDialog unavailable; deferred tabs were not built")
    from src.launchers.settings_dialog import TAB_APPEARANCE, TAB_LAYOUT

    settings_widget.show()
    settings_widget.sync_commit_model_caption(TAB_LAYOUT)
    assert settings_widget.btn_apply.isVisible() is False

    settings_widget.sync_commit_model_caption(TAB_APPEARANCE)
    assert settings_widget.btn_apply.isVisible() is True
    settings_widget.hide()
