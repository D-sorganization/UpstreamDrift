"""Plate declarations dialog for the ground-reaction tile (OF-07 follow-up, #13087).

Presentation only: one row per :data:`FORM_FIELDS` entry, grouped by section.
The texts are turned into a :class:`PlateImportContext` by
:func:`context_from_form`; a refused value is shown under the form and the
dialog stays open with that field focused.
"""

from __future__ import annotations

from collections.abc import Mapping

from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.theme.sidekick_tokens import get_current_sidekick_tokens
from src.tools.ground_reaction_plots.plate_import_form import (
    FORM_FIELDS,
    FormField,
    PlateFormError,
    PlateImportContext,
    context_from_form,
)

__all__ = ["PlateImportDialog"]

_SECTION_TITLES = {
    "Plate": "Plate Declarations",
    "Stance": "Stance Placement",
    "Clock": "Clock Alignment",
}


class PlateImportDialog(QDialog):
    """Ask for the facts a plate trial file cannot carry.

    ``texts`` prefills the rows (for example the values entered last time).
    """

    def __init__(
        self, parent: QWidget | None = None, texts: Mapping[str, str] | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Plate Trial Declarations")
        self._context: PlateImportContext | None = None
        self._editors: dict[str, QLineEdit | QComboBox] = {}
        root = QVBoxLayout(self)
        for section, title in _SECTION_TITLES.items():
            root.addWidget(self._section_box(section, title))
        self.error_label = QLabel("")
        error = get_current_sidekick_tokens()["sidekick.color.error"]
        self.error_label.setStyleSheet(f"color: {error};")
        self.error_label.setWordWrap(True)
        self.error_label.setVisible(False)
        root.addWidget(self.error_label)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self._prefill(texts or {})

    def _section_box(self, section: str, title: str) -> QGroupBox:
        box = QGroupBox(title)
        form = QFormLayout(box)
        for field in FORM_FIELDS:
            if field.section == section:
                editor = _editor(field)
                self._editors[field.key] = editor
                form.addRow(field.label, editor)
        return box

    def _prefill(self, texts: Mapping[str, str]) -> None:
        for key, text in texts.items():
            editor = self._editors.get(key)
            if editor is None:
                raise ValueError(f"{key}: unknown form field")
            if isinstance(editor, QComboBox):
                index = editor.findText(text)
                if index < 0:
                    raise ValueError(f"{key}: {text!r} is not an offered choice")
                editor.setCurrentIndex(index)
            else:
                editor.setText(text)

    def texts(self) -> dict[str, str]:
        """The current text of every row, keyed by form field."""
        return {
            key: editor.currentText()
            if isinstance(editor, QComboBox)
            else editor.text()
            for key, editor in self._editors.items()
        }

    def context(self) -> PlateImportContext | None:
        """The accepted context, or ``None`` before a successful accept."""
        return self._context

    def ask(self) -> PlateImportContext | None:
        """Run the dialog modally; the context, or ``None`` when cancelled."""
        if self.exec() != QDialog.DialogCode.Accepted:
            return None
        return self._context

    def accept(self) -> None:
        """Accept only when the form declares a valid context."""
        try:
            self._context = context_from_form(self.texts())
        except PlateFormError as exc:
            self._context = None
            self.error_label.setText(str(exc))
            self.error_label.setVisible(True)
            self._editors[exc.key].setFocus()
            return
        self.error_label.setVisible(False)
        super().accept()


def _editor(field: FormField) -> QLineEdit | QComboBox:
    if field.choices:
        combo = QComboBox()
        combo.addItems(field.choices)
        combo.setCurrentText(field.initial)
        return combo
    line = QLineEdit(field.initial)
    if field.required:
        line.setPlaceholderText("required")
    return line
