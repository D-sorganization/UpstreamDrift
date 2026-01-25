"""AI Assistant Panel for Golf Modeling Suite.

This module provides the main AI assistant conversation panel,
including message display, input handling, and streaming support.

The panel integrates with the selected AI provider and displays
responses with markdown rendering.
"""

from __future__ import annotations

# Python 3.10 compatibility: UTC was added in 3.11
from datetime import datetime, timezone

from src.shared.python.logging_config import get_logger

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc  # noqa: UP017
from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from shared.python.ai.adapters.base import BaseAgentAdapter
    from shared.python.ai.gui.settings_dialog import AISettings

from src.shared.python.ai.types import ConversationContext, ExpertiseLevel

logger = get_logger(__name__)


class MessageWidget(QFrame):
    """Widget displaying a single message in the conversation."""

    def __init__(
        self,
        role: str,
        content: str,
        timestamp: datetime | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """Initialize message widget.

        Args:
            role: Message role (user, assistant, system).
            content: Message content.
            timestamp: When the message was created.
            parent: Parent widget.
        """
        super().__init__(parent)
        self._role = role
        self._content = content
        self._timestamp = timestamp or datetime.now(UTC)
        self._setup_ui()
        self._apply_style()

    def _setup_ui(self) -> None:
        """Set up the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)

        # Header with role and time
        header = QHBoxLayout()

        role_label = QLabel(self._get_role_display())
        role_label.setStyleSheet("font-weight: bold;")
        header.addWidget(role_label)

        header.addStretch()

        time_label = QLabel(self._timestamp.strftime("%H:%M"))
        time_label.setStyleSheet("color: #888; font-size: 11px;")
        header.addWidget(time_label)

        layout.addLayout(header)

        # Content
        self._content_label = QTextEdit()
        self._content_label.setReadOnly(True)
        self._content_label.setFrameShape(QFrame.Shape.NoFrame)
        self._content_label.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._content_label.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._content_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )
        self._content_label.setMarkdown(self._content)

        # Auto-resize to content
        doc = self._content_label.document()
        if doc is not None:
            doc.contentsChanged.connect(self._adjust_height)
        self._adjust_height()

        layout.addWidget(self._content_label)

    def _get_role_display(self) -> str:
        """Get display name for role."""
        role_map = {
            "user": "You",
            "assistant": "AI Assistant",
            "system": "System",
            "tool": "Tool Result",
        }
        return role_map.get(self._role, self._role.title())

    def _apply_style(self) -> None:
        """Apply styling based on role."""
        if self._role == "user":
            self.setStyleSheet(
                """
                MessageWidget {
                    background-color: #e3f2fd;
                    border-radius: 8px;
                    margin-left: 40px;
                }
                """
            )
        elif self._role == "assistant":
            self.setStyleSheet(
                """
                MessageWidget {
                    background-color: #f5f5f5;
                    border-radius: 8px;
                    margin-right: 40px;
                }
                """
            )
        elif self._role == "system":
            self.setStyleSheet(
                """
                MessageWidget {
                    background-color: #fff3e0;
                    border-radius: 8px;
                }
                """
            )

    def _adjust_height(self) -> None:
        """Adjust height to fit content."""
        doc = self._content_label.document()
        if doc is not None:
            doc_height = doc.size().height()
            self._content_label.setFixedHeight(int(doc_height) + 10)

    def append_content(self, text: str) -> None:
        """Append content to the message (for streaming).

        Args:
            text: Text to append.
        """
        self._content += text
        self._content_label.setMarkdown(self._content)

    def get_content(self) -> str:
        """Get current content."""
        return self._content


class StreamWorker(QThread):
    """Worker thread for streaming AI responses."""

    chunk_received = pyqtSignal(str)  # Emits content chunk
    finished = pyqtSignal()  # Emits when complete
    error = pyqtSignal(str)  # Emits error message

    def __init__(
        self,
        adapter: BaseAgentAdapter,
        message: str,
        context: ConversationContext,
        tools: list[Any],
    ) -> None:
        """Initialize stream worker.

        Args:
            adapter: AI adapter to use.
            message: User message.
            context: Conversation context.
            tools: Available tools.
        """
        super().__init__()
        self._adapter = adapter
        self._message = message
        self._context = context
        self._tools = tools

    def run(self) -> None:
        """Execute streaming in background thread."""
        try:
            for chunk in self._adapter.stream_response(
                self._message,
                self._context,
                self._tools,
            ):
                if chunk.content:
                    self.chunk_received.emit(chunk.content)
        except Exception as e:
            logger.exception("Streaming error")
            self.error.emit(str(e))
        finally:
            self.finished.emit()


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

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize AI assistant panel.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)
        self._context = ConversationContext()
        self._adapter: BaseAgentAdapter | None = None
        self._current_worker: StreamWorker | None = None
        self._current_assistant_message: MessageWidget | None = None
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = self._create_header()
        layout.addWidget(header)

        # Splitter for messages and input
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Message area
        self._message_area = self._create_message_area()
        splitter.addWidget(self._message_area)

        # Input area
        input_widget = self._create_input_area()
        splitter.addWidget(input_widget)

        # Set splitter sizes (80% messages, 20% input)
        splitter.setSizes([400, 100])
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter)

    def _create_header(self) -> QWidget:
        """Create the panel header."""
        header = QFrame()
        header.setStyleSheet(
            """
            QFrame {
                background-color: #1976d2;
                padding: 8px;
            }
            QLabel {
                color: white;
            }
            QPushButton {
                background-color: transparent;
                color: white;
                border: 1px solid white;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.2);
            }
            """
        )

        layout = QHBoxLayout(header)

        # Title
        title = QLabel("🤖 AI Assistant")
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(title)

        # Status indicator
        self._status_label = QLabel("Ready")
        self._status_label.setStyleSheet("font-size: 11px;")
        layout.addWidget(self._status_label)

        layout.addStretch()

        # New chat button
        new_chat_btn = QPushButton("New Chat")
        new_chat_btn.clicked.connect(self._on_new_chat)
        layout.addWidget(new_chat_btn)

        # Settings button
        settings_btn = QPushButton("⚙️")
        settings_btn.setToolTip("Settings")
        settings_btn.clicked.connect(self.settings_requested.emit)
        layout.addWidget(settings_btn)

        return header

    def _create_message_area(self) -> QScrollArea:
        """Create the message display area."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Container for messages
        self._message_container = QWidget()
        self._message_layout = QVBoxLayout(self._message_container)
        self._message_layout.setContentsMargins(8, 8, 8, 8)
        self._message_layout.setSpacing(8)
        self._message_layout.addStretch()

        scroll.setWidget(self._message_container)

        # Add welcome message
        self._add_system_message(
            "👋 Welcome to the Golf Modeling Suite AI Assistant!\n\n"
            "I can help you:\n"
            "- Load and analyze C3D motion capture files\n"
            "- Run inverse dynamics simulations\n"
            "- Interpret joint torques and forces\n"
            "- Explain biomechanics concepts\n\n"
            "How can I help you today?"
        )

        return scroll

    def _create_input_area(self) -> QWidget:
        """Create the message input area."""
        widget = QFrame()
        widget.setStyleSheet(
            """
            QFrame {
                background-color: #fafafa;
                border-top: 1px solid #ddd;
            }
            """
        )

        layout = QVBoxLayout(widget)

        # Input text area
        self._input_edit = QPlainTextEdit()
        self._input_edit.setPlaceholderText(
            "Type your message here... (Ctrl+Enter to send)"
        )
        self._input_edit.setMaximumHeight(100)
        layout.addWidget(self._input_edit)

        # Buttons
        button_layout = QHBoxLayout()

        # Expertise level indicator
        self._expertise_label = QLabel("Level: Beginner")
        self._expertise_label.setStyleSheet("color: #666; font-size: 11px;")
        button_layout.addWidget(self._expertise_label)

        button_layout.addStretch()

        # Send button
        self._send_btn = QPushButton("Send")
        self._send_btn.setDefault(True)
        self._send_btn.clicked.connect(self._on_send)
        self._send_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #1976d2;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1565c0;
            }
            QPushButton:disabled {
                background-color: #ccc;
            }
            """
        )
        button_layout.addWidget(self._send_btn)

        layout.addLayout(button_layout)

        return widget

    def _connect_signals(self) -> None:
        """Connect internal signals."""
        # Ctrl+Enter to send
        shortcut = QShortcut(QKeySequence("Ctrl+Return"), self._input_edit)
        shortcut.activated.connect(self._on_send)

    def _on_send(self) -> None:
        """Handle send button click."""
        message = self._input_edit.toPlainText().strip()
        if not message:
            return

        # Clear input
        self._input_edit.clear()

        # Add user message
        self._add_message("user", message)

        # Emit signal
        self.message_sent.emit(message)

        # Process message if adapter is set
        if self._adapter:
            self._process_message(message)

    def _process_message(self, message: str) -> None:
        """Process a user message with the AI adapter.

        Args:
            message: User's message.
        """
        if not self._adapter:
            self._add_system_message(
                "⚠️ No AI provider configured. Click ⚙️ to set up a provider."
            )
            return

        # Update status
        self._set_status("Thinking...")
        self._send_btn.setEnabled(False)

        # Add context
        self._context.add_user_message(message)

        # Create streaming worker
        self._current_worker = StreamWorker(
            self._adapter,
            message,
            self._context,
            [],  # Tools will be added later
        )
        self._current_worker.chunk_received.connect(self._on_stream_chunk)
        self._current_worker.finished.connect(self._on_stream_finished)
        self._current_worker.error.connect(self._on_stream_error)

        # Create empty assistant message
        self._current_assistant_message = self._add_message("assistant", "")

        # Start streaming
        self._current_worker.start()

    def _on_stream_chunk(self, content: str) -> None:
        """Handle incoming stream chunk.

        Args:
            content: Content chunk.
        """
        if self._current_assistant_message:
            self._current_assistant_message.append_content(content)
            self._scroll_to_bottom()

    def _on_stream_finished(self) -> None:
        """Handle stream completion."""
        self._set_status("Ready")
        self._send_btn.setEnabled(True)

        # Add to context
        if self._current_assistant_message:
            self._context.add_assistant_message(
                self._current_assistant_message.get_content()
            )

        self._current_assistant_message = None
        self._current_worker = None

    def _on_stream_error(self, error: str) -> None:
        """Handle stream error.

        Args:
            error: Error message.
        """
        self._set_status("Error")
        self._send_btn.setEnabled(True)

        if self._current_assistant_message:
            self._current_assistant_message.append_content(f"\n\n⚠️ **Error:** {error}")

        self._current_assistant_message = None
        self._current_worker = None

    def _add_message(
        self,
        role: str,
        content: str,
        timestamp: datetime | None = None,
    ) -> MessageWidget:
        """Add a message to the conversation.

        Args:
            role: Message role.
            content: Message content.
            timestamp: Optional timestamp.

        Returns:
            The created MessageWidget.
        """
        # Insert before the stretch
        idx = self._message_layout.count() - 1

        widget = MessageWidget(role, content, timestamp)
        self._message_layout.insertWidget(idx, widget)

        self._scroll_to_bottom()
        return widget

    def _add_system_message(self, content: str) -> MessageWidget:
        """Add a system message.

        Args:
            content: Message content.

        Returns:
            The created MessageWidget.
        """
        return self._add_message("system", content)

    def _scroll_to_bottom(self) -> None:
        """Scroll message area to bottom."""
        # Guard against being called before _message_area is assigned
        if not hasattr(self, "_message_area"):
            return
        scroll = self._message_area
        scrollbar = scroll.verticalScrollBar()
        if scrollbar is not None:
            scrollbar.setValue(scrollbar.maximum())

    def _set_status(self, status: str) -> None:
        """Update status indicator.

        Args:
            status: Status text.
        """
        self._status_label.setText(status)

    def _on_new_chat(self) -> None:
        """Start a new chat session."""
        # Clear messages (except welcome)
        while self._message_layout.count() > 1:
            item = self._message_layout.takeAt(0)
            if item is not None:
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

        # Reset context
        self._context = ConversationContext()

        # Add welcome back
        self._add_system_message("🔄 New chat started. How can I help you?")

    def set_adapter(self, adapter: BaseAgentAdapter) -> None:
        """Set the AI adapter to use.

        Args:
            adapter: AI adapter instance.
        """
        self._adapter = adapter
        self._set_status("Ready")

    def set_expertise_level(self, level: ExpertiseLevel) -> None:
        """Set the user's expertise level.

        Args:
            level: Expertise level.
        """
        self._context.user_expertise = level
        level_names = {
            ExpertiseLevel.BEGINNER: "Beginner",
            ExpertiseLevel.INTERMEDIATE: "Intermediate",
            ExpertiseLevel.ADVANCED: "Advanced",
            ExpertiseLevel.EXPERT: "Expert",
        }
        self._expertise_label.setText(f"Level: {level_names[level]}")

    def apply_settings(self, settings: AISettings) -> None:
        """Apply settings from dialog.

        Args:
            settings: Settings to apply.
        """
        from shared.python.ai.gui.settings_dialog import AIProvider, get_api_key
        from shared.python.ai.types import ExpertiseLevel

        # Set expertise level
        level_map = {
            1: ExpertiseLevel.BEGINNER,
            2: ExpertiseLevel.INTERMEDIATE,
            3: ExpertiseLevel.ADVANCED,
            4: ExpertiseLevel.EXPERT,
        }
        self.set_expertise_level(
            level_map.get(settings.expertise_level, ExpertiseLevel.BEGINNER)
        )

        # Create adapter based on provider
        adapter: BaseAgentAdapter | None = None

        if settings.provider == AIProvider.OLLAMA:
            from shared.python.ai.adapters.ollama_adapter import OllamaAdapter

            adapter = OllamaAdapter(
                host=settings.ollama_host,
                model=settings.model,
            )
        elif settings.provider == AIProvider.OPENAI:
            api_key = get_api_key(AIProvider.OPENAI)
            if api_key:
                from shared.python.ai.adapters.openai_adapter import OpenAIAdapter

                adapter = OpenAIAdapter(
                    api_key=api_key,
                    model=settings.model,
                )
        elif settings.provider == AIProvider.ANTHROPIC:
            api_key = get_api_key(AIProvider.ANTHROPIC)
            if api_key:
                from shared.python.ai.adapters.anthropic_adapter import AnthropicAdapter

                adapter = AnthropicAdapter(
                    api_key=api_key,
                    model=settings.model,
                )

        if adapter:
            self.set_adapter(adapter)
            self._add_system_message(
                f"✓ Connected to {settings.provider.name} ({settings.model})"
            )
        else:
            self._add_system_message(
                f"⚠️ Could not connect to {settings.provider.name}. "
                "Please check your settings."
            )
