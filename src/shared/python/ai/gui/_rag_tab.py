"""RAG / Knowledge Base tab for the AI settings dialog (Tools #2762)."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.ai.access_policy import ChatAccessMode
from src.shared.python.theme.style_constants import Styles


class RagTab(QWidget):
    """RAG enable, access mode, and rebuild-index controls."""

    rebuild_index_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)

        rag_group = QGroupBox("Knowledge Base Settings")
        rag_layout = QVBoxLayout(rag_group)

        self.rag_enabled_check = QCheckBox("Enable Codebase Awareness (RAG)")
        self.rag_enabled_check.setToolTip(
            "Allow the AI to search your codebase and documents to answer questions."
        )
        rag_layout.addWidget(self.rag_enabled_check)

        self.access_mode_combo = QComboBox()
        self.access_mode_combo.addItem("No repo access", ChatAccessMode.NO_REPO_ACCESS)
        self.access_mode_combo.addItem(
            "Read-only diagnostics", ChatAccessMode.READ_ONLY_DIAGNOSTICS
        )
        self.access_mode_combo.addItem("Agent/tools", ChatAccessMode.AGENT_TOOLS)
        self.access_mode_combo.setToolTip(
            "Controls which codebase and local tools the assistant may receive."
        )
        rag_layout.addWidget(self.access_mode_combo)

        self.auto_index_check = QCheckBox("Auto-index codebase when chat opens")
        self.auto_index_check.setToolTip(
            "Rebuild the codemap each time the chat dock connects so the "
            "assistant has fresh symbol/import context. Slower first open."
        )
        rag_layout.addWidget(self.auto_index_check)

        layout.addWidget(rag_group)

        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group)

        rebuild_btn = QPushButton("Rebuild Knowledge Index")
        rebuild_btn.setToolTip("Scan the codebase and rebuild the search index.")
        rebuild_btn.clicked.connect(self._on_rebuild_index)
        actions_layout.addWidget(rebuild_btn)

        info_label = QLabel(
            "Rebuilding the index analyses your 'src' directory. "
            "This happens locally and no code is sent to the cloud during indexing."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet(Styles.TEXT_MUTED)
        actions_layout.addWidget(info_label)

        layout.addWidget(actions_group)
        layout.addStretch()

    def _on_rebuild_index(self) -> None:
        self.rebuild_index_requested.emit()
        QMessageBox.information(
            self,
            "Rebuild Started",
            "Index rebuild started in background. "
            "The assistant will be updated shortly.",
        )


__all__ = ["RagTab"]
