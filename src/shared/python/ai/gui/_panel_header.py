"""Panel header controller for the AI assistant.

Owns the header UI subtree (provider/model/mode combos, access-mode combo,
auto-index checkbox, action buttons) and emits Qt signals when the user
changes any of them. The parent panel observes the signals; the controller
holds no back-reference to the panel (LOD).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
)

from src.shared.python.ai.access_policy import ChatAccessMode
from src.shared.python.ai.gui._provider_registry_data import (
    AIProvider,
    populate_model_combo,
    populate_provider_combo,
)

if TYPE_CHECKING:
    from src.shared.python.ai._settings_model import AISettings


_CHAT_MODES = (
    ("Ask", "ask"),
    ("Diagnose (read-only)", "diagnose"),
    ("Agent", "agent"),
)


class PanelHeaderController(QFrame):
    """Encapsulates the AI assistant's header strip.

    Signals are emitted on user interaction. The owning panel wires them
    to its own slots; controllers do not call methods on the panel directly.
    """

    provider_changed = pyqtSignal(object)  # AIProvider
    model_changed = pyqtSignal(str)
    mode_changed = pyqtSignal(str)
    access_mode_changed = pyqtSignal(object)  # ChatAccessMode
    auto_index_toggled = pyqtSignal(bool)
    new_chat_requested = pyqtSignal()
    peer_review_requested = pyqtSignal()
    condense_requested = pyqtSignal()
    show_full_history_requested = pyqtSignal()
    settings_requested = pyqtSignal()
    close_requested = pyqtSignal()
    copy_thread_requested = pyqtSignal()
    save_thread_requested = pyqtSignal()

    def __init__(self, initial_settings: AISettings, parent: Any = None) -> None:
        super().__init__(parent)
        self._syncing = False
        self._build(initial_settings)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build(self, settings: AISettings) -> None:
        layout = QHBoxLayout(self)
        self._add_title_widgets(layout, settings)
        self._add_mode_and_status(layout, settings)
        layout.addStretch()
        self._add_action_buttons(layout)

    def _add_title_widgets(self, layout: QHBoxLayout, settings: AISettings) -> None:
        self.provider_icon = QLabel("\U0001f916")
        layout.addWidget(self.provider_icon)

        self.provider_combo = QComboBox()
        self.provider_combo.setObjectName("aiProviderCombo")
        self.provider_combo.setToolTip("Select AI provider")
        populate_provider_combo(self.provider_combo)
        self._set_combo_data(self.provider_combo, settings.provider)
        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        layout.addWidget(self.provider_combo)

        self.model_combo = QComboBox()
        self.model_combo.setObjectName("aiModelCombo")
        self.model_combo.setToolTip("Select AI model")
        populate_model_combo(self.model_combo, settings.provider, settings.model)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        layout.addWidget(self.model_combo)

        self.model_label = QLabel("AI Assistant")
        self.model_label.setVisible(False)

        layout.addSpacing(10)

    def _add_mode_and_status(self, layout: QHBoxLayout, settings: AISettings) -> None:
        self.mode_combo = QComboBox()
        self.mode_combo.setObjectName("aiModeCombo")
        for label, value in _CHAT_MODES:
            self.mode_combo.addItem(label, value)
        self._set_combo_data(self.mode_combo, settings.chat_mode)
        self.mode_combo.setToolTip(
            "Select AI mode: Ask, Diagnose (read-only), or Agent"
        )
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        layout.addWidget(self.mode_combo)

        self.access_mode_combo = QComboBox()
        self.access_mode_combo.addItem("No repo access", ChatAccessMode.NO_REPO_ACCESS)
        self.access_mode_combo.addItem(
            "Read-only diagnostics", ChatAccessMode.READ_ONLY_DIAGNOSTICS
        )
        self.access_mode_combo.addItem("Agent/tools", ChatAccessMode.AGENT_TOOLS)
        self.access_mode_combo.setToolTip(
            "Controls which repo and local tools the assistant may receive."
        )
        self.access_mode_combo.currentIndexChanged.connect(self._on_access_mode_changed)
        layout.addWidget(self.access_mode_combo)

        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

    def _add_action_buttons(self, layout: QHBoxLayout) -> None:
        self.auto_index_checkbox = QCheckBox("Auto-Index")
        self.auto_index_checkbox.setToolTip(
            "Rebuild the local codebase index when chat opens."
        )
        self.auto_index_checkbox.toggled.connect(self.auto_index_toggled.emit)
        layout.addWidget(self.auto_index_checkbox)

        copy_thread_btn = QPushButton("Copy Thread")
        copy_thread_btn.setToolTip("Copy full conversation to clipboard")
        copy_thread_btn.clicked.connect(self.copy_thread_requested.emit)
        layout.addWidget(copy_thread_btn)

        save_thread_btn = QPushButton("Save as Markdown")
        save_thread_btn.setToolTip("Save conversation to a .md file")
        save_thread_btn.clicked.connect(self.save_thread_requested.emit)
        layout.addWidget(save_thread_btn)

        self.token_count_label = QLabel("~0 tokens")
        self.token_count_label.setObjectName("aiTokenCountLabel")
        self.token_count_label.setToolTip(
            "Estimated token count for the current active thread."
        )
        layout.addWidget(self.token_count_label)

        self.condense_btn = QPushButton("Condense")
        self.condense_btn.setObjectName("aiCondenseBtn")
        self.condense_btn.setToolTip(
            "Summarise earlier messages to free context space. "
            "Raw history is preserved for undo."
        )
        self.condense_btn.clicked.connect(self.condense_requested.emit)
        layout.addWidget(self.condense_btn)

        self.show_history_btn = QPushButton("Full History")
        self.show_history_btn.setObjectName("aiShowHistoryBtn")
        self.show_history_btn.setToolTip(
            "Toggle between condensed and full message history."
        )
        self.show_history_btn.setVisible(False)
        self.show_history_btn.clicked.connect(self.show_full_history_requested.emit)
        layout.addWidget(self.show_history_btn)

        new_chat_btn = QPushButton("New Chat")
        new_chat_btn.clicked.connect(self.new_chat_requested.emit)
        layout.addWidget(new_chat_btn)

        peer_review_btn = QPushButton("🔍 Peer Review")
        peer_review_btn.setObjectName("peerReviewBtn")
        peer_review_btn.setToolTip(
            "Request a second AI agent to critically review this conversation"
        )
        peer_review_btn.clicked.connect(self.peer_review_requested.emit)
        layout.addWidget(peer_review_btn)

        settings_btn = QPushButton("⚙️")
        settings_btn.setToolTip("Settings")
        settings_btn.clicked.connect(self.settings_requested.emit)
        layout.addWidget(settings_btn)

        close_btn = QPushButton("✕")
        close_btn.setToolTip("Close AI Chat")
        close_btn.clicked.connect(self.close_requested.emit)
        layout.addWidget(close_btn)

    # ------------------------------------------------------------------
    # Sync / public API
    # ------------------------------------------------------------------
    @staticmethod
    def _set_combo_data(combo: QComboBox, data: object) -> bool:
        idx = combo.findData(data)
        if idx < 0:
            return False
        combo.setCurrentIndex(idx)
        return True

    def sync_controls(self, settings: AISettings) -> None:
        """Repopulate combos to reflect ``settings`` without firing signals."""
        self._syncing = True
        try:
            self.provider_combo.blockSignals(True)
            self.model_combo.blockSignals(True)
            self.mode_combo.blockSignals(True)
            self._set_combo_data(self.provider_combo, settings.provider)
            populate_model_combo(self.model_combo, settings.provider, settings.model)
            self._set_combo_data(self.mode_combo, settings.chat_mode)
        finally:
            self.provider_combo.blockSignals(False)
            self.model_combo.blockSignals(False)
            self.mode_combo.blockSignals(False)
            self._syncing = False

    def sync_access_mode(self, mode: ChatAccessMode) -> None:
        for i in range(self.access_mode_combo.count()):
            if self.access_mode_combo.itemData(i) == mode:
                self.access_mode_combo.blockSignals(True)
                self.access_mode_combo.setCurrentIndex(i)
                self.access_mode_combo.blockSignals(False)
                return

    def set_token_count(self, count: int) -> None:
        """Update the estimated token count display in the toolbar.

        Args:
            count: Estimated token count for the active thread.
        """
        self.token_count_label.setText(f"~{count:,} tokens")

    def set_condensed_mode(self, condensed: bool) -> None:
        """Show or hide the 'Full History' toggle button.

        Args:
            condensed: True when the thread is in condensed state.
        """
        self.show_history_btn.setVisible(condensed)

    def set_auto_index_checked(self, checked: bool) -> None:
        self.auto_index_checkbox.blockSignals(True)
        self.auto_index_checkbox.setChecked(checked)
        self.auto_index_checkbox.blockSignals(False)

    def auto_index_enabled(self) -> bool:
        return bool(self.auto_index_checkbox.isChecked())

    def update_models(self, models: list[str]) -> None:
        if not models:
            return
        current = self.model_combo.currentText()
        self.model_combo.blockSignals(True)
        try:
            self.model_combo.clear()
            for name in models:
                self.model_combo.addItem(name)
            idx = self.model_combo.findText(current)
            if idx < 0 and self.model_combo.count():
                idx = 0
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
        finally:
            self.model_combo.blockSignals(False)

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def set_provider_icon(self, icon: str) -> None:
        self.provider_icon.setText(icon)

    def set_model_label(self, text: str, tooltip: str = "") -> None:
        self.model_label.setText(text)
        if tooltip:
            self.model_label.setToolTip(tooltip)

    def apply_theme(self, colors: dict) -> None:
        """Restyle every header child from a flat theme color dict."""
        bg_primary = colors["bg_primary"]
        bg_alt = colors["bg_alt"]
        text_primary = colors["text_primary"]
        text_muted = colors["text_muted"]
        border = colors["border"]
        accent = colors["accent"]
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {bg_alt};
                padding: 10px;
                border-bottom: 1px solid {border};
            }}
            QLabel {{ color: {text_primary}; }}
            QPushButton {{
                background-color: transparent;
                color: {text_muted};
                border: 1px solid {border};
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {accent};
                color: #ffffff;
                border-color: {accent};
            }}
            """)
        combo_qss = f"""
            QComboBox {{
                background-color: {bg_primary};
                color: {text_primary};
                border: 1px solid {border};
                border-radius: 6px;
                padding: 4px 12px;
                min-width: 80px;
            }}
            QComboBox::drop-down {{ border: none; width: 20px; }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 4px solid {text_muted};
                margin-top: 2px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {bg_alt};
                color: {text_primary};
                border: 1px solid {border};
                border-radius: 4px;
                selection-background-color: {accent};
            }}
        """
        for combo in (self.provider_combo, self.model_combo, self.mode_combo):
            combo.setStyleSheet(combo_qss)
        self.status_label.setStyleSheet(
            f"font-size: 11px; color: {text_muted}; background: transparent;"
        )
        self.provider_icon.setStyleSheet(
            f"font-size: 18px; color: {text_primary}; background: transparent;"
        )
        self.model_label.setStyleSheet(
            f"font-size: 14px; font-weight: bold; color: {text_primary}; "
            "background: transparent;"
        )
        self.auto_index_checkbox.setStyleSheet(f"color: {text_primary};")

    @property
    def is_syncing(self) -> bool:
        return self._syncing

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _on_provider_changed(self, index: int) -> None:
        if self._syncing:
            return
        provider = self.provider_combo.itemData(index)
        if not isinstance(provider, AIProvider):
            return
        current_model = self.model_combo.currentText()
        self._syncing = True
        try:
            self.model_combo.blockSignals(True)
            populate_model_combo(self.model_combo, provider, current_model)
        finally:
            self.model_combo.blockSignals(False)
            self._syncing = False
        self.provider_changed.emit(provider)

    def _on_model_changed(self, index: int) -> None:
        if self._syncing or index < 0:
            return
        self.model_changed.emit(self.model_combo.currentText())

    def _on_mode_changed(self, index: int) -> None:
        if self._syncing or index < 0:
            return
        mode = self.mode_combo.currentData()
        if isinstance(mode, str):
            self.mode_changed.emit(mode)

    def _on_access_mode_changed(self, _index: int) -> None:
        from src.shared.python.ai.access_policy import coerce_access_mode

        mode = coerce_access_mode(self.access_mode_combo.currentData())
        self.access_mode_changed.emit(mode)
