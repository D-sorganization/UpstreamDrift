"""Sidebar widget for chat history navigation."""

from datetime import datetime
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.ai.gui.session_manager import ChatSessionManager
from src.shared.python.theme.style_constants import Styles


class SessionListWidgetItem(QListWidgetItem):
    """Custom list item to hold session metadata."""

    def __init__(self, session_data: dict[str, Any]) -> None:
        super().__init__()
        self.session_data = session_data

        # Format the text
        title = session_data.get("title", "New Chat")
        dt = session_data.get("timestamp")

        if dt and dt != datetime.min:
            time_str = dt.strftime("%Y-%m-%d %H:%M")
        else:
            time_str = "Unknown"

        self.setText(f"{title}\n{time_str}")
        self.setToolTip(session_data.get("snippet", ""))


class ChatHistorySidebar(QWidget):
    """Sidebar for managing chat sessions (multi-conversation, archive, delete)."""

    session_selected = pyqtSignal(str)  # Emits session ID
    new_chat_requested = pyqtSignal()

    def __init__(
        self, session_manager: ChatSessionManager, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._manager = session_manager
        self._manager.sessions_updated.connect(self.refresh_lists)
        self.refresh_theme()
        self._setup_ui()
        self.refresh_lists()

    def refresh_theme(self) -> None:
        """Dynamically update colors based on the current theme."""
        try:
            from src.shared.python.theme.theme_manager import get_theme_manager

            colors = get_theme_manager().get_current_colors()
            bg_base = colors.get("bg", "#1e1e1e")
            bg_alt = colors.get("group_bg", "#252526")
            text_color = colors.get("text", "#e0e0e0")
            text_muted = colors.get("text_secondary", "#888888")
            accent = colors.get("accent", "#FF8800")
            border = colors.get("border", "#3c3c3c")
        except ImportError:
            bg_base = "#1e1e1e"
            bg_alt = "#252526"
            text_color = "#e0e0e0"
            text_muted = "#888888"
            accent = "#FF8800"
            border = "#3c3c3c"

        self._theme_colors = {
            "bg_base": bg_base,
            "bg_alt": bg_alt,
            "text_color": text_color,
            "text_muted": text_muted,
            "accent": accent,
            "border": border,
        }

        if hasattr(self, "_header"):
            self._header.setStyleSheet(f"""
                QFrame {{
                    background-color: {self._theme_colors["bg_alt"]};
                    border-bottom: 1px solid {self._theme_colors["border"]};
                }}
            """)
        if hasattr(self, "_title_label"):
            self._title_label.setStyleSheet(
                f"font-weight: bold; color: {self._theme_colors['text_color']};"
            )
        if hasattr(self, "_new_btn"):
            self._new_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: transparent;
                    color: {self._theme_colors["accent"]};
                    border: 1px solid {self._theme_colors["accent"]};
                    border-radius: 4px;
                    padding: 4px 8px;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: {self._theme_colors["accent"]};
                    color: {self._theme_colors["bg_base"]};
                }}
            """)
        if hasattr(self, "_tabs"):
            self._tabs.setStyleSheet(f"""
                QTabWidget::pane {{ border: none; }}
                QTabBar::tab {{
                    background: {self._theme_colors["bg_alt"]};
                    color: {self._theme_colors["text_muted"]};
                    padding: 6px 12px;
                    border: 1px solid {self._theme_colors["border"]};
                    border-bottom: none;
                    border-top-left-radius: 4px;
                    border-top-right-radius: 4px;
                }}
                QTabBar::tab:selected {{
                    background: {self._theme_colors["bg_base"]};
                    color: {self._theme_colors["accent"]};
                    font-weight: bold;
                }}
            """)

        if hasattr(self, "_active_list"):
            self._apply_list_style(self._active_list)
        if hasattr(self, "_archive_list"):
            self._apply_list_style(self._archive_list)

    def _apply_list_style(self, list_widget: QListWidget) -> None:
        list_widget.setStyleSheet(f"""
            QListWidget {{
                background-color: {self._theme_colors["bg_base"]};
                color: {self._theme_colors["text_color"]};
                border: none;
                outline: 0;
            }}
            QListWidget::item {{
                border-bottom: 1px solid {self._theme_colors["border"]};
                padding: 8px;
            }}
            QListWidget::item:selected {{
                background-color: transparent;
                border-left: 3px solid {self._theme_colors["accent"]};
            }}
            QListWidget::item:hover:!selected {{
                background-color: {self._theme_colors["bg_alt"]};
            }}
        """)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header with New Chat button
        self._header = QFrame()
        header_layout = QHBoxLayout(self._header)
        header_layout.setContentsMargins(8, 8, 8, 8)

        self._title_label = QLabel("Conversations")
        header_layout.addWidget(self._title_label)

        header_layout.addStretch()

        self._new_btn = QPushButton("⊕ New")
        self._new_btn.setToolTip("Start a new conversation")
        self._new_btn.clicked.connect(self.new_chat_requested.emit)
        header_layout.addWidget(self._new_btn)

        layout.addWidget(self._header)

        # Tabs for Active / Archived
        self._tabs = QTabWidget()

        # Active List
        self._active_list = QListWidget()
        self._setup_list_widget(self._active_list)
        self._tabs.addTab(self._active_list, "Active")

        # Archived List
        self._archive_list = QListWidget()
        self._setup_list_widget(self._archive_list)
        self._tabs.addTab(self._archive_list, "Archived")

        layout.addWidget(self._tabs)

    def _setup_list_widget(self, list_widget: QListWidget) -> None:
        self._apply_list_style(list_widget)
        list_widget.setWordWrap(True)
        list_widget.itemClicked.connect(self._on_item_clicked)
        # Context menu for right click
        list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        list_widget.customContextMenuRequested.connect(
            lambda pos, lw=list_widget: self._show_context_menu(pos, lw)
        )

    def refresh_lists(self) -> None:
        """Reload sessions from the manager and update UI."""
        self._active_list.clear()
        self._archive_list.clear()

        sessions = self._manager.list_sessions()
        for s in sessions:
            item = SessionListWidgetItem(s)
            if s.get("archived", False):
                self._archive_list.addItem(item)
            else:
                self._active_list.addItem(item)

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        if isinstance(item, SessionListWidgetItem):
            self.session_selected.emit(item.session_data["id"])

    def _show_context_menu(self, pos: Any, list_widget: QListWidget) -> None:
        item = list_widget.itemAt(pos)
        if not item or not isinstance(item, SessionListWidgetItem):
            return

        menu = QMenu(self)
        session_id = item.session_data["id"]
        is_archived = item.session_data.get("archived", False)

        if is_archived:
            restore_action = menu.addAction("Restore")
            restore_action.triggered.connect(
                lambda: self._manager.archive_session(session_id, False)
            )
        else:
            archive_action = menu.addAction("Archive")
            archive_action.triggered.connect(
                lambda: self._manager.archive_session(session_id, True)
            )

        menu.addSeparator()
        delete_action = menu.addAction("Delete Permanently")
        delete_action.triggered.connect(lambda: self._confirm_delete(session_id))

        menu.exec(list_widget.mapToGlobal(pos))

    def _confirm_delete(self, session_id: str) -> None:
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            "Are you sure you want to permanently delete this conversation?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._manager.delete_session(session_id)
