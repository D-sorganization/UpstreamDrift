"""One commit model and one close contract for the Settings dialog.

Issue #8896 (with #8895). The Settings dialog mixes two commit models in a
single ``QTabWidget``:

* **Live-apply** -- Layout, Configuration, Diagnostics, MCP Servers,
  Processes. A control change calls the launcher immediately.
* **Deferred** -- Appearance, Startup, Notifications, Performance. These
  tabs are lifted out of a never-shown ``PreferencesDialog`` and commit
  only when "Apply Preferences" is pressed.

Nothing in the UI said which was which, the Apply button was visible (and
meaningless) on the live-apply tabs, and the footer was a bare Close with
no dirty check -- so changing the theme and font size and pressing Close
discarded both silently, while the same gesture on the Layout tab
persisted instantly.

This mixin supplies the missing half:

* a caption naming the current tab's commit model;
* dirty tracking for the deferred tabs;
* an Apply button that is visible only where it means something and
  marked when there is something to apply;
* a close contract that prompts Save / Discard / Cancel, and that also
  runs the running-build guard.

**Why signal-based dirty tracking.** The deferred tabs are built by
``src/shared/python/ui/preferences_dialog.py``, which is mid-retirement and
out of bounds for this change. Rather than snapshot fields this module
would have to know about, it walks the tab widget trees once and connects
whatever value-changed signal each control has. That needs no knowledge of
``PreferencesDialog``'s field set and does not break when it gains one.
"""

from __future__ import annotations

from typing import Any, Final

from PyQt6.QtWidgets import QLabel, QMessageBox, QPushButton, QWidget

from src.launchers.build_close_guard import confirm_cancel_running_build_for_close

__all__ = [
    "APPLY_LABEL",
    "APPLY_LABEL_DIRTY",
    "CAPTION_DEFERRED",
    "CAPTION_LIVE",
    "SettingsCloseContract",
]

APPLY_LABEL: Final[str] = "Apply Preferences"
APPLY_LABEL_DIRTY: Final[str] = "Apply Preferences *"

CAPTION_LIVE: Final[str] = "Changes on this tab apply immediately."
CAPTION_DEFERRED: Final[str] = (
    "Changes on this tab are not saved until you press Apply Preferences."
)

#: Value-changed signals, most specific first. The first one a widget has is
#: connected and the rest skipped, so a QComboBox does not report twice.
_DIRTY_SIGNALS: Final[tuple[str, ...]] = (
    "currentIndexChanged",
    "valueChanged",
    "stateChanged",
    "toggled",
    "textChanged",
)


class SettingsCloseContract:
    """Commit-model captioning, preference dirty tracking, and close guards.

    Mixed into ``SettingsWidget``. Requires the host to provide ``tabs``,
    ``commit_model_label``, ``btn_apply`` (optional -- absent when the
    preferences import fails), ``_prefs_dialog``, ``preference_tab_indexes``,
    and optionally ``build_thread``.
    """

    #: Overridden by the host with the real tab indexes.
    preference_tab_indexes: frozenset[int] = frozenset()

    # ---- construction ---------------------------------------------------

    def install_commit_model_caption(self, layout: Any) -> None:
        """Add the caption that names the current tab's commit model."""
        self.commit_model_label = QLabel("")
        self.commit_model_label.setObjectName("settings-commit-model")
        self.commit_model_label.setWordWrap(True)
        layout.addWidget(self.commit_model_label)

    def install_apply_button(self, layout: Any) -> None:
        """Add the Apply Preferences button and start dirty tracking.

        The button is hidden on the live-apply tabs, where it was visible
        and meaningless before this change.
        """
        self.btn_apply = QPushButton(APPLY_LABEL)
        self.btn_apply.clicked.connect(self.apply_preferences)
        layout.addWidget(self.btn_apply)
        self.wire_preference_dirty_tracking()

    # ---- dirty tracking ------------------------------------------------

    def wire_preference_dirty_tracking(self) -> None:
        """Connect every control on the deferred tabs to the dirty flag."""
        self._prefs_dirty = False
        for index in self.preference_tab_indexes:
            widget = self.tabs.widget(index)  # type: ignore[attr-defined]
            if widget is None:
                continue
            for child in widget.findChildren(QWidget):
                for signal_name in _DIRTY_SIGNALS:
                    signal = getattr(child, signal_name, None)
                    if signal is None:
                        continue
                    try:
                        signal.connect(self._mark_preferences_dirty)
                    except (AttributeError, TypeError):
                        continue
                    break

    def _mark_preferences_dirty(self, *_args: Any) -> None:
        """Record an uncommitted preference edit and reflect it in the UI."""
        if getattr(self, "_prefs_dirty", False):
            return
        self._prefs_dirty = True
        self.refresh_apply_button()

    def is_preferences_dirty(self) -> bool:
        """Return True when a deferred tab has uncommitted edits."""
        return bool(getattr(self, "_prefs_dirty", False))

    # ---- apply + caption -----------------------------------------------

    def refresh_apply_button(self) -> None:
        """Mark and enable the Apply button according to the dirty flag."""
        button = getattr(self, "btn_apply", None)
        if button is None:
            return
        dirty = self.is_preferences_dirty()
        button.setText(APPLY_LABEL_DIRTY if dirty else APPLY_LABEL)
        button.setEnabled(dirty)

    def apply_preferences(self) -> None:
        """Commit the deferred tabs and clear the dirty flag."""
        self._prefs_dialog._on_apply()  # type: ignore[attr-defined]
        self._prefs_dirty = False
        self.refresh_apply_button()

    def sync_commit_model_caption(self, index: int) -> None:
        """Show the current tab's commit model, and the Apply button with it."""
        deferred = index in self.preference_tab_indexes
        label = getattr(self, "commit_model_label", None)
        if label is not None:
            label.setText(CAPTION_DEFERRED if deferred else CAPTION_LIVE)
        button = getattr(self, "btn_apply", None)
        if button is not None:
            button.setVisible(deferred)
        self.refresh_apply_button()

    # ---- close contract -------------------------------------------------

    def confirm_discard_preferences(self) -> bool:
        """Prompt Save / Discard / Cancel for uncommitted preference edits.

        Returns True when the caller may proceed with closing. A Save
        answer applies the preferences first and proceeds only if that
        cleared the flag, so choosing Save can never lose them.
        """
        if not self.is_preferences_dirty():
            return True
        answer = QMessageBox.question(
            self,  # type: ignore[arg-type]
            "Unsaved Preferences",
            "You have preference changes that have not been applied. "
            "Save them before closing?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Discard:
            return True
        if answer != QMessageBox.StandardButton.Save:
            return False
        self.apply_preferences()
        return not self.is_preferences_dirty()

    def confirm_close(self, event: Any) -> bool:
        """Return True when the settings surface may close.

        One entry point for both guards, called by this widget's own
        ``closeEvent`` **and** by ``SettingsDialog.closeEvent``: a QDialog
        does not deliver close events to its children, so before this
        change the build guard never ran on the dialog's own close paths.
        """
        if not confirm_cancel_running_build_for_close(
            self,  # type: ignore[arg-type]
            event,
            getattr(self, "build_thread", None),
            log_message="Cancelling Docker build before closing settings",
        ):
            return False
        if not self.confirm_discard_preferences():
            event.ignore()
            return False
        return True
