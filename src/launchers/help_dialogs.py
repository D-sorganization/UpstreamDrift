"""Help and layout management dialogs for the launcher.

Provides HelpDialog, LayoutManagerDialog, and ContextHelpDock.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.logging_pkg.logging_config import get_logger

from .startup import ASSETS_DIR, REPOS_ROOT

logger = get_logger(__name__)


class HelpDialog(QDialog):
    """Dialog to display help documentation."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Golf Suite - Help")
        self.resize(800, 600)
        layout = QVBoxLayout(self)
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        layout.addWidget(self.text_area)

        help_path = ASSETS_DIR / "help.md"
        if help_path.exists():
            self.text_area.setMarkdown(help_path.read_text(encoding="utf-8"))
        else:
            self.text_area.setText("Help file not found.")

        btn = QPushButton("Close")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)


class LayoutManagerDialog(QDialog):
    """Dialog allowing users to add or remove launcher tiles."""

    def __init__(
        self,
        available_models: dict[str, Any],
        active_models: list[str],
        parent: QWidget | None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Customize Launcher Tiles")
        self.resize(520, 520)

        layout = QVBoxLayout(self)

        description = QLabel(
            "Select which applications should appear on the launcher grid. "
            "Checked items will be visible while unchecked items will be hidden."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        self.list_widget = QListWidget()

        sorted_models = sorted(
            available_models.values(),
            key=lambda model: getattr(model, "name", "").lower(),
        )

        for model in sorted_models:
            item = QListWidgetItem(f"{model.name} — {model.description}")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked
                if model.id in active_models
                else Qt.CheckState.Unchecked
            )
            item.setData(Qt.ItemDataRole.UserRole, model.id)
            self.list_widget.addItem(item)

        layout.addWidget(self.list_widget)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_ids(self) -> list[str]:
        """Return IDs of all checked models."""
        selections: list[str] = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item and item.checkState() == Qt.CheckState.Checked:
                model_id = item.data(Qt.ItemDataRole.UserRole)
                if model_id:
                    selections.append(str(model_id))
        return selections


class ContextHelpDock(QDockWidget):
    """Context-aware help drawer."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Quick Help", parent)
        self.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
        )

        # Content
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.text_area.setStyleSheet(
            "background-color: #252526; color: #cccccc; border: none; padding: 10px;"
        )
        self.setWidget(self.text_area)

        # Default content
        self.update_context(None)

    def update_context(self, model_id: str | None) -> None:
        """Update help content based on selected model."""
        if not model_id:
            self.text_area.setMarkdown(
                "### Context Aware Help\n\nSelect a model to view its documentation and quick start guide."
            )
            return

        # Map ID to doc file
        doc_file = self._get_doc_file(model_id)
        if doc_file and doc_file.exists():
            try:
                content = doc_file.read_text(encoding="utf-8")
                self.text_area.setMarkdown(content)
            except (RuntimeError, ValueError, OSError) as e:
                self.text_area.setText(f"Failed to load documentation: {e}")
        else:
            self.text_area.setMarkdown(
                f"### {model_id}\n\nNo specific documentation available."
            )

    def _get_doc_file(self, model_id: str) -> Path | None:
        docs_dir = REPOS_ROOT / "docs" / "engines"

        if "mujoco" in model_id:
            return docs_dir / "mujoco.md"
        if "drake" in model_id:
            return docs_dir / "drake.md"
        if "pinocchio" in model_id:
            return docs_dir / "pinocchio.md"
        if "matlab" in model_id:
            return docs_dir / "matlab.md"
        if "urdf" in model_id:
            return REPOS_ROOT / "tools" / "urdf_generator" / "README.md"

        return None
