"""Shared native dialog controls for Capture Rig editors."""

from PyQt6.QtWidgets import QDialog, QDialogButtonBox


def save_cancel_buttons(dialog: QDialog) -> QDialogButtonBox:
    """Use the platform button order and each editor's validation handlers."""
    buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel,
        parent=dialog,
    )
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    return buttons
