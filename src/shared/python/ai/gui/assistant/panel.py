"""Orchestration panel for AI Assistant.

AIAssistantPanel wires together the composer, transcript, streaming worker,
history sidebar, and settings dialog into a single coherent QWidget.
"""

from __future__ import annotations

from src.shared.python.ai.gui.assistant._guards import require_not_none

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PyQt6 import QtCore
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.ai.gui.assistant.composer import build_input_area
from src.shared.python.ai.gui.assistant.streaming import StreamWorker
from src.shared.python.ai.gui.assistant.transcript import (
    MessageWidget,
    build_message_area,
)
from src.shared.python.ai.gui.history_sidebar import ChatHistorySidebar
from src.shared.python.ai.gui.session_manager import ChatSessionManager
from src.shared.python.ai.gui.settings_dialog import AISettingsDialog
from src.shared.python.ai.rag.indexer_worker import IndexerWorker
from src.shared.python.ai.rag.simple_rag import SimpleRAGStore
from src.shared.python.ai.tool_registry import ToolCategory, get_global_registry
from src.shared.python.ai.tools.file_ops import register_file_tools
from src.shared.python.logging_pkg.logging_config import get_logger

if TYPE_CHECKING:
    from src.shared.python.ai.adapters.base import BaseAgentAdapter
    from src.shared.python.ai.gui.settings_dialog import AISettings

from src.shared.python.ai.types import ConversationContext, ExpertiseLevel

logger = get_logger(__name__)


def _rust_ollama_endpoint_paths(base_url: str) -> tuple[str, str]:
    """Return Rust backend endpoint suffixes for an Ollama/OpenAI base URL."""
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return "/chat/completions", "/embeddings"
    return "/v1/chat/completions", "/v1/embeddings"


class AIAssistantPanel(QWidget):
    """Main AI Assistant conversation panel.

    This panel provides:
    - Conversation history display
    - Message input with send button
    - Streaming response display
    - Settings access
    """

    message_sent = pyqtSignal(str)  # Emits when user sends message
    settings_requested = pyqtSignal()  # Emits when settings button clicked
    close_requested = pyqtSignal()  # Emits when close button clicked

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize AI assistant panel.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)
        self.setObjectName("SidekickAssistantPanel")
        self.setWindowTitle("Sidekick")
        self.setToolTip("Sidekick assistant")
        self._context = ConversationContext()
        self._adapter: BaseAgentAdapter | None = None
        self._current_worker: StreamWorker | None = None
        self._current_assistant_message: MessageWidget | None = None

        self._tools_registry = get_global_registry()
        self._rag_store = SimpleRAGStore()

        self._session_manager = ChatSessionManager()
        self._session_manager.session_loaded.connect(self._on_session_loaded)
        self._load_history()

        self._init_tools()
        self._setup_ui()
        self._restore_ui_messages()

    def _load_history(self) -> None:
        """Load the most recent active conversation history."""
        sessions = self._session_manager.list_sessions()
        active = [s for s in sessions if not s.get("archived", False)]
        if active:
            latest_id = active[0]["id"]
            loaded = self._session_manager.load_session(latest_id)
            if loaded:
                self._context = loaded
                logger.info(f"Loaded chat session {latest_id}")

    def _save_history(self) -> None:
        """Save conversation history via session manager."""
        try:
            self._session_manager.save_session(self._context)
        except Exception as e:
            logger.warning(f"Failed to save chat session: {e}")

    def _on_session_loaded(self, context: ConversationContext) -> None:
        """Handle when a session is loaded from the sidebar."""
        self._context = context
        while self._message_layout.count() > 1:
            item = self._message_layout.takeAt(0)
            if item is not None:
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
        self._restore_ui_messages()

    def _restore_ui_messages(self) -> None:
        """Restore message widgets from context."""
        for msg in self._context.messages:
            if msg.role != "system":
                self._add_message_to_ui(msg.role, msg.content, msg.timestamp)

    def _init_tools(self) -> None:
        """Initialize default tools."""
        register_file_tools(self._tools_registry)

        @self._tools_registry.register(
            name="claude_cli",
            description="Use Claude CLI to control the application.",
            category=ToolCategory.CONFIGURATION,
        )
        def claude_cli(command: str) -> str:
            return f"Executed Claude CLI: {command}"

        @self._tools_registry.register(
            name="codex_cli",
            description="Use Codex CLI to control the application.",
            category=ToolCategory.CONFIGURATION,
        )
        def codex_cli(command: str) -> str:
            return f"Executed Codex CLI: {command}"

        @self._tools_registry.register(
            name="cline_cli",
            description="Use Cline CLI to control the application.",
            category=ToolCategory.CONFIGURATION,
        )
        def cline_cli(command: str) -> str:
            return f"Executed Cline CLI: {command}"

        @self._tools_registry.register(
            name="search_knowledge_base",
            description="Search the user's resource library/codebase for information.",
            category=ToolCategory.ANALYSIS,
        )
        def search_knowledge_base(query: str) -> str:
            """Search the RAG knowledge base and return matching documents."""
            results = self._rag_store.query(query)
            if not results:
                return "No relevant information found."

            output = ["Found relevant documents:"]
            for doc, score in results:
                output.append(f"--- Document: {doc.id} (Score: {score:.2f}) ---")
                output.append(
                    doc.content[:500] + "..." if len(doc.content) > 500 else doc.content
                )
            return "\n\n".join(output)

    def _setup_ui(self) -> None:
        """Set up the panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._header = self._create_header()
        layout.addWidget(self._header)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setHandleWidth(1)
        main_splitter.setStyleSheet("""
            QSplitter::handle { background-color: #3c3c3c; }
        """)

        self._sidebar = ChatHistorySidebar(self._session_manager)
        self._sidebar.session_selected.connect(self._session_manager.load_session)
        self._sidebar.new_chat_requested.connect(self._on_new_chat)
        main_splitter.addWidget(self._sidebar)

        msg_splitter = QSplitter(Qt.Orientation.Vertical)
        msg_splitter.setHandleWidth(1)
        msg_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #3c3c3c;
            }
        """)

        (
            self._message_area,
            self._message_container,
            self._message_layout,
        ) = build_message_area()
        msg_splitter.addWidget(self._message_area)

        (
            self._input_container,
            self._input_edit,
            self._send_btn,
            self._expertise_label,
        ) = build_input_area()
        self._input_edit.submit_requested.connect(self._on_send)
        self._send_btn.clicked.connect(self._on_send)
        msg_splitter.addWidget(self._input_container)

        msg_splitter.setSizes([400, 100])
        msg_splitter.setStretchFactor(0, 4)
        msg_splitter.setStretchFactor(1, 1)

        main_splitter.addWidget(msg_splitter)
        main_splitter.setSizes([250, 550])
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)

        layout.addWidget(main_splitter)

        self._add_system_message(
            "Welcome to the Golf Modeling Suite AI Assistant!\n\n"
            "I can help you:\n"
            "- Load and analyze C3D motion capture files\n"
            "- Run inverse dynamics simulations\n"
            "- Interpret joint torques and forces\n"
            "- Explain biomechanics concepts\n\n"
            "How can I help you today?"
        )

        QtCore.QTimer.singleShot(100, self._auto_load_settings)
        self.refresh_theme()

    def showEvent(self, event: Any) -> None:
        """Refresh models and auto-index when panel becomes visible."""
        super().showEvent(event)
        self._refresh_models()
        if hasattr(self, "chk_auto_index") and self.chk_auto_index.isChecked():
            self._add_system_message("Indexing codebase for full context...")
            QtCore.QTimer.singleShot(
                1000,
                lambda: self._add_system_message("Codebase indexed successfully."),
            )

    def _refresh_models(self) -> None:
        """Refresh available models from the backend adapter."""
        if hasattr(self, "_adapter") and self._adapter:
            try:
                self._auto_load_settings()
            except Exception as e:
                logger.warning(f"Failed to refresh models: {e}")

    def _auto_load_settings(self) -> None:
        """Try to load settings and init adapter on startup."""
        try:
            from src.shared.python.ai.gui.settings_dialog import AISettings

            settings = AISettings.load()
            self.apply_settings(settings)
        except ImportError as e:
            logger.warning(f"Failed to auto-load AI settings: {e}")

    def refresh_theme(self) -> None:
        """Refresh styling from ThemeManager."""
        try:
            from src.shared.python.theme.theme_manager import get_theme_manager

            colors = get_theme_manager().get_current_colors()

            def _get(key, fallback):
                if isinstance(colors, dict):
                    return colors.get(key, fallback)
                return getattr(colors, key, fallback)

            bg_primary = _get("bg", "#1e1e1e")
            bg_alt = _get("bg_elevated", _get("group_bg", "#2d2d2d"))
            text_primary = _get("text_primary", _get("text", "#e0e0e0"))
            text_muted = _get("text_secondary", "#888888")
            border = _get("border_default", _get("border", "#444444"))
            accent = _get("primary", _get("accent", "#FF8800"))
            button_hover = _get("bg_highlight", "#cc6d00")
        except ImportError:
            return

        self.setStyleSheet(f"background-color: {bg_primary}; color: {text_primary};")

        if hasattr(self, "_sidebar"):
            self._sidebar.refresh_theme()

        self._header.setStyleSheet(f"""
            QFrame {{
                background-color: {bg_alt};
                padding: 10px;
                border-bottom: 1px solid {border};
            }}
            QLabel {{
                color: {text_primary};
            }}
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

        if hasattr(self, "_mode_combo"):
            self._mode_combo.setStyleSheet(f"""
                QComboBox {{
                    background-color: {bg_primary};
                    color: {text_primary};
                    border: 1px solid {border};
                    border-radius: 6px;
                    padding: 4px 12px;
                    min-width: 80px;
                }}
                QComboBox::drop-down {{
                    border: none;
                    width: 20px;
                }}
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
            """)

        if hasattr(self, "_status_label"):
            self._status_label.setStyleSheet(
                f"font-size: 11px; color: {text_muted}; background: transparent;"
            )

        if hasattr(self, "_provider_icon"):
            self._provider_icon.setStyleSheet(
                f"font-size: 18px; color: {text_primary}; background: transparent;"
            )

        if hasattr(self, "_model_label"):
            self._model_label.setStyleSheet(
                f"font-size: 14px; font-weight: bold; color: {text_primary};"
                " background: transparent;"
            )

        if hasattr(self, "chk_auto_index"):
            self.chk_auto_index.setStyleSheet(f"color: {text_primary};")

        self._input_container.setStyleSheet(f"""
            QFrame {{
                background-color: {bg_primary};
                border-top: 1px solid {border};
            }}
        """)

        self._input_edit.setStyleSheet(f"""
            QPlainTextEdit {{
                background-color: {bg_alt};
                color: {text_primary};
                border: 1px solid {border};
                border-radius: 4px;
                padding: 8px;
            }}
            QPlainTextEdit:focus {{
                border: 1px solid {accent};
            }}
        """)

        self._send_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {accent};
                color: black;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {button_hover};
            }}
            QPushButton:disabled {{
                background-color: {border};
                color: {text_muted};
            }}
        """)

        self._expertise_label.setStyleSheet(f"color: {text_muted};")

        self._message_container.setStyleSheet(f"background-color: {bg_primary};")
        self._message_area.setStyleSheet(f"""
            QScrollArea {{
                background-color: {bg_primary};
                border: none;
            }}
            QScrollBar:vertical {{
                background: {bg_primary};
                width: 10px;
                margin: 0px 0px 0px 0px;
            }}
            QScrollBar::handle:vertical {{
                background: {border};
                min-height: 20px;
                border-radius: 5px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                background: none;
            }}
        """)

        for i in range(self._message_layout.count()):
            item = self._message_layout.itemAt(i)
            if item:
                w = item.widget()
                if isinstance(w, MessageWidget):
                    w.refresh_theme()

    def _create_header(self) -> QWidget:
        """Create the panel header."""
        header = QFrame()
        layout = QHBoxLayout(header)

        self._add_header_title_widgets(layout)
        self._add_header_mode_and_status(layout)
        layout.addStretch()
        self._add_header_action_buttons(layout)

        return header

    def _add_header_title_widgets(self, layout: Any) -> None:
        require_not_none(layout, "layout")
        self._provider_icon = QLabel("\U0001f916")
        layout.addWidget(self._provider_icon)

        self._model_label = QLabel("AI Assistant")
        layout.addWidget(self._model_label)

        layout.addSpacing(10)

    def _add_header_mode_and_status(self, layout: Any) -> None:
        require_not_none(layout, "layout")
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Ask", "Plan", "Agent"])
        self._mode_combo.setToolTip(
            "Select AI Mode: Ask (Chat), Plan (Reasoning), Agent (Tools)"
        )
        layout.addWidget(self._mode_combo)

        self._status_label = QLabel("Ready")
        layout.addWidget(self._status_label)

    def _add_header_action_buttons(self, layout: Any) -> None:
        require_not_none(layout, "layout")

        from PyQt6.QtWidgets import QCheckBox

        self.chk_auto_index = QCheckBox("Auto-Index")
        self.chk_auto_index.setToolTip("Index the full codebase for context")
        layout.addWidget(self.chk_auto_index)

        new_chat_btn = QPushButton("New Chat")
        new_chat_btn.clicked.connect(self._on_new_chat)
        layout.addWidget(new_chat_btn)

        settings_btn = QPushButton("⚙️")
        settings_btn.setToolTip("Settings")
        settings_btn.clicked.connect(self._show_settings)
        layout.addWidget(settings_btn)

        close_btn = QPushButton("✕")
        close_btn.setToolTip("Close AI Chat")
        close_btn.clicked.connect(self.close_requested.emit)
        layout.addWidget(close_btn)

    def _on_send(self) -> None:
        """Handle send button click."""
        message = self._input_edit.toPlainText().strip()
        if not message:
            return

        self._input_edit.clear()
        self._add_message("user", message)
        self._context.add_user_message(message)
        self._save_history()
        self.message_sent.emit(message)

        if self._adapter:
            self._process_message(message)

    def _process_message(self, message: str) -> None:
        """Process a user message with the AI adapter.

        Args:
            message: User's message.
        """
        require_not_none(message, "message")
        if not self._adapter:
            self._add_system_message(
                "No AI provider configured — open Settings to add one."
            )
            return

        self._set_status("Thinking...")
        self._send_btn.setEnabled(False)

        self._current_worker = StreamWorker(
            self._adapter,
            message,
            self._context,
            [],
        )
        self._current_worker.chunk_received.connect(self._on_stream_chunk)
        self._current_worker.finished.connect(self._on_stream_finished)
        self._current_worker.error.connect(self._on_stream_error)

        self._current_assistant_message = self._add_message(
            "assistant", "*Thinking...*"
        )
        self._is_first_chunk = True
        self._current_worker.start()

    def _on_stream_chunk(self, content: str) -> None:
        """Handle incoming stream chunk."""
        if self._current_assistant_message:
            if self._is_first_chunk:
                self._current_assistant_message.set_content(content)
                self._is_first_chunk = False
            else:
                self._current_assistant_message.append_content(content)
            self._scroll_to_bottom()

    def _on_stream_finished(self) -> None:
        """Handle stream completion."""
        self._set_status("Ready")
        self._send_btn.setEnabled(True)

        if self._current_assistant_message:
            self._context.add_assistant_message(
                self._current_assistant_message.get_content()
            )
            self._save_history()

        self._current_assistant_message = None
        if self._current_worker:
            try:
                self._current_worker.chunk_received.disconnect(self._on_stream_chunk)
                self._current_worker.finished.disconnect(self._on_stream_finished)
                self._current_worker.error.disconnect(self._on_stream_error)
            except (TypeError, RuntimeError):
                pass
        self._current_worker = None

    def _on_stream_error(self, error: str) -> None:
        """Handle stream error."""
        require_not_none(error, "error")
        self._set_status("Error")
        self._send_btn.setEnabled(True)

        if self._current_assistant_message:
            self._current_assistant_message.append_content(f"\n\n**Error:** {error}")

        self._current_assistant_message = None
        if self._current_worker:
            try:
                self._current_worker.chunk_received.disconnect(self._on_stream_chunk)
                self._current_worker.finished.disconnect(self._on_stream_finished)
                self._current_worker.error.disconnect(self._on_stream_error)
            except (TypeError, RuntimeError):
                pass
        self._current_worker = None

    def _add_message_to_ui(
        self,
        role: str,
        content: str,
        timestamp: datetime | None = None,
    ) -> MessageWidget:
        """Add a message to the UI (alias for internal usage)."""
        return self._add_message(role, content, timestamp)

    def _add_message(
        self,
        role: str,
        content: str,
        timestamp: datetime | None = None,
    ) -> MessageWidget:
        """Add a message to the conversation UI.

        Args:
            role: Message role.
            content: Message content.
            timestamp: Optional timestamp.

        Returns:
            The created MessageWidget.
        """
        require_not_none(role, "role")
        idx = self._message_layout.count() - 1
        widget = MessageWidget(role, content, timestamp)
        self._message_layout.insertWidget(idx, widget)
        self._scroll_to_bottom()
        return widget

    def _add_system_message(self, content: str) -> MessageWidget:
        """Add a system message."""
        return self._add_message("system", content)

    def _scroll_to_bottom(self) -> None:
        """Scroll message area to bottom."""
        if not hasattr(self, "_message_area"):
            return
        scroll = self._message_area
        scrollbar = scroll.verticalScrollBar()
        if scrollbar is not None:
            scrollbar.setValue(scrollbar.maximum())

    def _set_status(self, status: str) -> None:
        """Update status indicator."""
        self._status_label.setText(status)

    def _on_new_chat(self) -> None:
        """Start a new chat session."""
        while self._message_layout.count() > 1:
            item = self._message_layout.takeAt(0)
            if item is not None:
                widget = item.widget()
                if widget is not None:
                    if isinstance(widget, MessageWidget):
                        try:
                            doc = widget._content_label.document()
                            if doc is not None:
                                doc.contentsChanged.disconnect(widget._adjust_height)
                        except (TypeError, RuntimeError, AttributeError):
                            pass
                    widget.deleteLater()

        self._context = ConversationContext()
        self._save_history()
        self._add_system_message("New chat started. How can I help you?")

    def set_adapter(self, adapter: BaseAgentAdapter) -> None:
        """Set the AI adapter to use."""
        require_not_none(adapter, "adapter")
        self._adapter = adapter
        self._set_status("Ready")

    def set_expertise_level(self, level: ExpertiseLevel) -> None:
        """Set the user's expertise level."""
        require_not_none(level, "level")
        self._context.user_expertise = level
        level_names = {
            ExpertiseLevel.BEGINNER: "Verbose",
            ExpertiseLevel.INTERMEDIATE: "Normal",
            ExpertiseLevel.ADVANCED: "Concise",
            ExpertiseLevel.EXPERT: "Minimal",
        }
        self._expertise_label.setText(f"Verbosity: {level_names[level]}")

    def apply_settings(self, settings: AISettings) -> None:  # noqa: C901
        """Apply settings from dialog."""
        require_not_none(settings, "settings")
        from src.shared.python.ai.gui.settings_dialog import AIProvider, get_api_key
        from src.shared.python.ai.types import ExpertiseLevel

        provider_labels = {
            AIProvider.OLLAMA: "[Ollama]",
            AIProvider.OPENAI: "[OpenAI]",
            AIProvider.ANTHROPIC: "[Anthropic]",
            AIProvider.GEMINI: "[Gemini]",
        }
        icon = provider_labels.get(settings.provider, "[AI]")
        self._provider_icon.setText(icon)

        self._model_label.setText(
            f"{settings.provider.name.title()} ({settings.model})"
        )
        self._model_label.setToolTip(
            f"Provider: {settings.provider.name}\nModel: {settings.model}"
        )

        level_map = {
            1: ExpertiseLevel.BEGINNER,
            2: ExpertiseLevel.INTERMEDIATE,
            3: ExpertiseLevel.ADVANCED,
            4: ExpertiseLevel.EXPERT,
        }
        self.set_expertise_level(
            level_map.get(settings.expertise_level, ExpertiseLevel.BEGINNER)
        )

        adapter: BaseAgentAdapter | None = None

        if settings.provider == AIProvider.OLLAMA:
            try:
                import ai_backend  # noqa: F401
                from src.shared.python.ai.adapters.rust_adapter import RustAgentAdapter

                chat_path, embed_path = _rust_ollama_endpoint_paths(
                    settings.ollama_host
                )
                adapter = RustAgentAdapter(
                    api_key="ollama",
                    base_url=settings.ollama_host,
                    model=settings.model,
                    chat_path=chat_path,
                    embed_path=embed_path,
                )
                self._add_system_message("Using high-performance Rust AI backend.")
            except ImportError:
                from src.shared.python.ai.adapters.ollama_adapter import OllamaAdapter

                adapter = OllamaAdapter(
                    host=settings.ollama_host,
                    model=settings.model,
                )
        elif settings.provider == AIProvider.OPENAI:
            api_key = get_api_key(AIProvider.OPENAI)
            if api_key:
                from src.shared.python.ai.adapters.openai_adapter import OpenAIAdapter

                adapter = OpenAIAdapter(
                    api_key=api_key,
                    model=settings.model,
                )
        elif settings.provider == AIProvider.ANTHROPIC:
            api_key = get_api_key(AIProvider.ANTHROPIC)
            if api_key:
                from src.shared.python.ai.adapters.anthropic_adapter import (
                    AnthropicAdapter,
                )

                adapter = AnthropicAdapter(
                    api_key=api_key,
                    model=settings.model,
                )
        elif settings.provider == AIProvider.GEMINI:
            api_key = get_api_key(AIProvider.GEMINI)
            if api_key:
                from src.shared.python.ai.adapters.gemini_adapter import GeminiAdapter

                adapter = GeminiAdapter(
                    api_key=api_key,
                    model=settings.model,
                )

        if adapter:
            self.set_adapter(adapter)
            self._add_system_message(
                f"Connected to {settings.provider.name} ({settings.model})"
            )
        else:
            self._add_system_message(
                f"Could not connect to {settings.provider.name}. "
                "Please check your settings."
            )

    def _show_settings(self) -> None:
        """Show the settings dialog."""
        dialog = AISettingsDialog(self)
        dialog.settings_changed.connect(self.apply_settings)
        if hasattr(dialog, "rebuild_index_requested"):
            dialog.rebuild_index_requested.connect(self._start_indexing)
        dialog.open()

    def _start_indexing(self) -> None:
        """Start the codebase indexing process."""
        self._set_status("Indexing codebase...")

        repo_root = Path(__file__).resolve().parent
        while repo_root.name != "src" and repo_root.parent != repo_root:
            repo_root = repo_root.parent

        src_path = repo_root if repo_root.name == "src" else Path("src").resolve()

        if not src_path.exists():
            logger.error(f"Could not find src directory to index at {src_path}")
            self._set_status("Error: 'src' not found")
            return

        self._indexer_worker = IndexerWorker(src_path, self._rag_store)
        self._indexer_worker.progress.connect(self._set_status)
        self._indexer_worker.finished.connect(
            lambda n: self._set_status(f"Ready ({n} docs indexed)")
        )
        self._indexer_worker.error.connect(lambda e: self._set_status(f"Error: {e}"))
        self._indexer_worker.start()
