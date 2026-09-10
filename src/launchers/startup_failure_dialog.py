"""Actionable startup failure dialog for the launcher splash (issue #8360).

Replaces the old ``QMessageBox.critical`` + ``QApplication.quit()`` path:
when a required startup phase fails, or the splash watchdog fires, the user
gets a deterministic, keyboard-accessible choice instead of a dead app.
"""

from __future__ import annotations

from enum import Enum

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

__all__ = ["StartupFailureAction", "StartupFailureDialog"]


class StartupFailureAction(str, Enum):
    """Closed set of user decisions after a startup failure."""

    RETRY = "retry"
    CONTINUE = "continue"
    CLOSE = "close"


class StartupFailureDialog(QDialog):
    """Modal dialog offering Retry / Continue / Copy diagnostics / Close.

    Invariants:
        * ``action`` is always a :class:`StartupFailureAction`; it defaults
          to ``CLOSE`` so dismissing the dialog (Escape, window close) never
          silently continues.
        * Every button has a mnemonic and is reachable with Tab; Enter
          activates the default *Continue* button and Escape maps to Close.
        * *Copy diagnostics* never closes the dialog.
    """

    def __init__(
        self,
        message: str,
        diagnostics: str,
        *,
        parent: QWidget | None = None,
    ) -> None:
        if not message:
            raise ValueError("message must be non-empty")
        if not isinstance(diagnostics, str):
            raise TypeError("diagnostics must be a string")
        super().__init__(parent)
        self.action = StartupFailureAction.CLOSE
        self._diagnostics = diagnostics

        self.setWindowTitle("UpstreamDrift startup problem")
        self.setModal(True)
        self.setMinimumWidth(560)

        layout = QVBoxLayout(self)
        summary = QLabel(message, self)
        summary.setWordWrap(True)
        summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(summary)

        self.diagnostics_view = QPlainTextEdit(self)
        self.diagnostics_view.setReadOnly(True)
        self.diagnostics_view.setPlainText(diagnostics)
        self.diagnostics_view.setAccessibleName("Startup diagnostics")
        layout.addWidget(self.diagnostics_view)

        buttons = QDialogButtonBox(self)
        self.retry_button = QPushButton("&Retry startup", self)
        self.continue_button = QPushButton("&Continue without provider", self)
        self.copy_button = QPushButton("Copy &diagnostics", self)
        self.close_button = QPushButton("C&lose UpstreamDrift", self)
        role = QDialogButtonBox.ButtonRole
        buttons.addButton(self.retry_button, role.ResetRole)
        buttons.addButton(self.continue_button, role.AcceptRole)
        buttons.addButton(self.copy_button, role.ActionRole)
        buttons.addButton(self.close_button, role.RejectRole)
        layout.addWidget(buttons)

        self.continue_button.setDefault(True)
        self.retry_button.clicked.connect(
            lambda: self._choose(StartupFailureAction.RETRY)
        )
        self.continue_button.clicked.connect(
            lambda: self._choose(StartupFailureAction.CONTINUE)
        )
        self.copy_button.clicked.connect(self.copy_diagnostics)
        self.close_button.clicked.connect(
            lambda: self._choose(StartupFailureAction.CLOSE)
        )

    @property
    def diagnostics(self) -> str:
        """Diagnostics text shown in (and copied from) the dialog."""
        return self._diagnostics

    def copy_diagnostics(self) -> None:
        """Copy the diagnostics text to the system clipboard."""
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(self._diagnostics)

    def _choose(self, action: StartupFailureAction) -> None:
        self.action = action
        self.done(int(action != StartupFailureAction.CLOSE))

    def reject(self) -> None:
        """Escape / window close map to the Close action."""
        self.action = StartupFailureAction.CLOSE
        super().reject()

    def ask(self) -> StartupFailureAction:
        """Run the dialog modally and return the chosen action."""
        self.exec()
        return self.action
