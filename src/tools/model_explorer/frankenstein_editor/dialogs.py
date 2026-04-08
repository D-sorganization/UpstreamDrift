from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)


class StealComponentDialog(QDialog):
    """Dialog for configuring component stealing with renaming."""

    def __init__(
        self,
        comp_type: str,
        original_name: str,
        parent: QWidget | None = None,
    ) -> None:
        """Initialize the dialog."""
        if not (comp_type is not None):
            raise ValueError("comp_type must be provided")
        super().__init__(parent)
        self.setWindowTitle("Copy Component")
        self.setMinimumWidth(350)

        layout = QVBoxLayout(self)

        # Info
        layout.addWidget(QLabel(f"Copying {comp_type}: {original_name}"))

        # Name input
        form = QFormLayout()
        self.name_edit = QLineEdit(original_name)
        form.addRow("New name:", self.name_edit)

        # Prefix option
        self.prefix_edit = QLineEdit()
        self.prefix_edit.setPlaceholderText("e.g., 'imported_'")
        form.addRow("Add prefix:", self.prefix_edit)

        layout.addLayout(form)

        # Include related checkbox (for links)
        if comp_type == "link":
            self.include_materials = QLabel(
                "Note: Referenced materials will also be copied"
            )
            layout.addWidget(self.include_materials)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_new_name(self) -> str:
        """Get the new name with prefix."""
        prefix = self.prefix_edit.text()
        name = self.name_edit.text()
        return prefix + name
