# ruff: noqa: E501
"""UI-construction helper for :class:`ChatDockWidget`.

Extracts the ~340-line body of ``ChatDockWidget._setup_ui`` into a free
function so the parent module fits in the repo's 1500-line budget.

The builder returns an explicit ``ChatDockView`` containing every widget and
action it creates. ``ChatDockWidget`` mirrors those fields to the historical
``_foo`` aliases for one compatibility cycle, but this module no longer has to
scatter private attribute writes across the widget implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .._thinking_indicator import ThinkingIndicator
from . import ai_dropdowns
from .history_sidebar import HistorySidebar
from .input import install_enter_submit
from .queue_panel import QueuePanel
from .styling import get_theme_colors

if TYPE_CHECKING:
    pass


@dataclass
class ChatDockView:
    """Concrete UI state created by ``build_chat_dock_ui``.

    Field names intentionally omit the legacy leading underscore. The mirror
    helper maps ``send_btn`` to ``dock._send_btn`` and so on for downstream
    compatibility while keeping the creation contract in one typed place.
    """

    status_label: QLabel | None = None
    tools_btn: QPushButton | None = None
    tools_menu: QMenu | None = None
    action_copy_thread: QAction | None = None
    action_export_markdown: QAction | None = None
    action_export_text: QAction | None = None
    action_export_html: QAction | None = None
    action_condense_keep_recent: QAction | None = None
    action_condense_semantic: QAction | None = None
    action_condense_pinned: QAction | None = None
    action_export_thread: QAction | None = None
    action_condense_thread: QAction | None = None
    action_request_review: QAction | None = None
    action_manage_memory: QAction | None = None
    token_indicator: QLabel | None = None
    auto_condense_threshold: int = 8000
    new_chat_btn: QPushButton | None = None
    toggle_history_btn: QPushButton | None = None
    splitter: QSplitter | None = None
    history_sidebar: HistorySidebar | None = None
    mode_combo: QComboBox | None = None
    shell_combo: QComboBox | None = None
    provider_combo: QComboBox | None = None
    ai_provider_combo: QComboBox | None = None
    ai_model_combo: QComboBox | None = None
    ai_thinking_combo: QComboBox | None = None
    terminal_start_btn: QPushButton | None = None
    terminal_stop_btn: QPushButton | None = None
    scroll_area: QScrollArea | None = None
    message_container: QWidget | None = None
    message_layout: QVBoxLayout | None = None
    terminal_output: QPlainTextEdit | None = None
    content_stack: QStackedWidget | None = None
    thinking_indicator: ThinkingIndicator | None = None
    queue_panel: QueuePanel | None = None
    input_edit: QPlainTextEdit | None = None
    upload_btn: QPushButton | None = None
    screenshot_btn: QPushButton | None = None
    mic_btn: QPushButton | None = None
    agent_mode_combo: QComboBox | None = None
    send_btn: QPushButton | None = None
    steer_btn: QPushButton | None = None
    stop_agent_btn: QPushButton | None = None


def mirror_chat_dock_view(dock: Any, view: ChatDockView) -> None:
    """Mirror ``ChatDockView`` fields to legacy ``dock._foo`` aliases."""
    for field_info in fields(view):
        setattr(dock, f"_{field_info.name}", getattr(view, field_info.name))


def build_chat_dock_ui(dock: Any) -> ChatDockView:
    """Build the full chat dock widget UI for ``dock``.

    DbC:
        Pre: ``dock`` must be a ``QDockWidget`` subclass with all
        configuration attributes already set (``_accent_color``,
        ``_placeholder_text``, ``_theme_provider``, etc.).
        Post: returned ``ChatDockView`` has every created widget/action field
        populated; ``dock.setWidget(...)`` has been called.
    """
    view = ChatDockView()
    colors = get_theme_colors(dock._theme_provider)
    bg_primary = colors.get("bg", "#1e1e1e")
    bg_alt = colors.get("group_bg", "#2d2d2d")
    text_primary = colors.get("text", "#e0e0e0")
    text_secondary = colors.get("text_secondary", "#888")
    border = colors.get("border", "#444")
    button_hover = colors.get("button_hover", "#ffaa33")

    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(4, 4, 4, 4)
    layout.setSpacing(4)

    # Status bar
    status_row = QHBoxLayout()
    view.status_label = QLabel("Connecting...")
    view.status_label.setStyleSheet(f"color: {text_secondary}; font-size: 10px;")
    status_row.addWidget(view.status_label, stretch=1)

    view.tools_btn = QPushButton("Tools")
    view.tools_btn.setToolTip("Chat tools and actions")
    view.tools_menu = QMenu(dock)
    view.action_copy_thread = view.tools_menu.addAction("Copy Entire Thread")
    export_menu = view.tools_menu.addMenu("Export Thread")
    view.action_export_markdown = (
        export_menu.addAction("Markdown...") if export_menu is not None else None
    )
    view.action_export_text = (
        export_menu.addAction("Plain Text...") if export_menu is not None else None
    )
    view.action_export_html = (
        export_menu.addAction("HTML...") if export_menu is not None else None
    )
    condense_menu = view.tools_menu.addMenu("Condense Thread")
    view.action_condense_keep_recent = (
        condense_menu.addAction("Keep recent...") if condense_menu is not None else None
    )
    view.action_condense_semantic = (
        condense_menu.addAction("Semantic summary...")
        if condense_menu is not None
        else None
    )
    view.action_condense_pinned = (
        condense_menu.addAction("Pinned anchor...")
        if condense_menu is not None
        else None
    )
    # Backwards-compat alias for tests that introspect _action_export_thread.
    view.action_export_thread = view.action_export_markdown
    view.action_condense_thread = view.action_condense_keep_recent
    view.action_request_review = view.tools_menu.addAction("Request Agent Review...")
    view.action_manage_memory = view.tools_menu.addAction("Manage Memory...")
    if view.action_manage_memory is not None:
        view.action_manage_memory.triggered.connect(dock.open_memory_panel)
    if view.action_copy_thread is not None:
        view.action_copy_thread.triggered.connect(dock._copy_entire_thread)
    if view.action_export_markdown is not None:
        view.action_export_markdown.triggered.connect(
            lambda: dock._export_thread("markdown", "Markdown Files (*.md)", ".md")
        )
    if view.action_export_text is not None:
        view.action_export_text.triggered.connect(
            lambda: dock._export_thread("text", "Text Files (*.txt)", ".txt")
        )
    if view.action_export_html is not None:
        view.action_export_html.triggered.connect(
            lambda: dock._export_thread("html", "HTML Files (*.html)", ".html")
        )
    if view.action_condense_keep_recent is not None:
        view.action_condense_keep_recent.triggered.connect(
            lambda: dock._run_condense_local("keep_recent")
        )
    if view.action_condense_semantic is not None:
        view.action_condense_semantic.triggered.connect(
            lambda: dock._run_condense_local("semantic_summary")
        )
    if view.action_condense_pinned is not None:
        view.action_condense_pinned.triggered.connect(
            lambda: dock._run_condense_local("pinned_anchor")
        )
    if view.action_request_review is not None:
        view.action_request_review.triggered.connect(dock._request_review)
    view.tools_btn.setMenu(view.tools_menu)

    # Token-budget indicator + condense-now button.
    view.token_indicator = QLabel("0 tok")
    view.token_indicator.setToolTip(
        "Approximate token count for the current thread. "
        "When it exceeds the auto-condense threshold the thread will "
        "be condensed automatically."
    )
    view.token_indicator.setStyleSheet(f"color: {text_secondary}; font-size: 10px;")
    status_row.addWidget(view.token_indicator)
    view.auto_condense_threshold = 8000

    # New Chat and History buttons
    view.new_chat_btn = QPushButton("New Chat")
    view.new_chat_btn.setToolTip("Start a new chat conversation")
    view.new_chat_btn.setStyleSheet(
        "QPushButton {"
        f"  background-color: {bg_alt}; color: {text_primary};"
        "  border-radius: 4px; padding: 2px 6px; font-size: 11px;"
        "}"
        f"QPushButton:hover {{ background-color: {border}; }}"
    )
    view.new_chat_btn.clicked.connect(dock._on_new_chat_clicked)
    status_row.addWidget(view.new_chat_btn)

    view.toggle_history_btn = QPushButton("History")
    view.toggle_history_btn.setToolTip("Toggle conversation history sidebar")
    view.toggle_history_btn.setStyleSheet(
        "QPushButton {"
        f"  background-color: {bg_alt}; color: {text_primary};"
        "  border-radius: 4px; padding: 2px 6px; font-size: 11px;"
        "}"
        f"QPushButton:hover {{ background-color: {border}; }}"
    )
    view.toggle_history_btn.clicked.connect(dock._on_toggle_history)
    status_row.addWidget(view.toggle_history_btn)

    layout.addLayout(status_row)

    # QSplitter wrapping HistorySidebar and the main chat pane
    view.splitter = QSplitter(Qt.Orientation.Horizontal)
    view.splitter.setHandleWidth(1)

    view.history_sidebar = HistorySidebar(dock._session_manager, parent=dock)
    view.history_sidebar.setVisible(False)
    view.splitter.addWidget(view.history_sidebar)

    chat_pane = QWidget()
    chat_layout = QVBoxLayout(chat_pane)
    chat_layout.setContentsMargins(0, 0, 0, 0)
    chat_layout.setSpacing(4)
    view.splitter.addWidget(chat_pane)

    layout.addWidget(view.splitter, stretch=1)

    mode_row = QHBoxLayout()
    ai_dropdowns.build_ai_dropdowns(
        view,
        mode_row,
        on_combo_changed=dock._on_ai_combo_changed,
        refresh_model_combo=lambda: ai_dropdowns.refresh_ai_model_combo(
            view, dock._get_active_ai_adapter()
        ),
        refresh_thinking_combo=lambda: ai_dropdowns.refresh_ai_thinking_combo(
            view, dock._get_active_ai_adapter()
        ),
        sync_dropdowns=lambda: ai_dropdowns.sync_ai_dropdowns(
            view,
            current_provider=dock._current_provider,
            current_model=dock._current_model,
            current_thinking_level=dock._current_thinking_level,
        ),
    )

    view.mode_combo = QComboBox()
    view.mode_combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
    view.mode_combo.setMinimumWidth(0)
    view.mode_combo.addItem("Chat", "chat")
    view.mode_combo.addItem("Terminal", "terminal")
    view.mode_combo.currentIndexChanged.connect(dock._on_mode_changed)
    mode_row.addWidget(view.mode_combo)

    view.shell_combo = QComboBox()
    view.shell_combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
    view.shell_combo.setMinimumWidth(0)
    view.shell_combo.currentIndexChanged.connect(dock._on_terminal_shell_changed)
    mode_row.addWidget(view.shell_combo)

    view.provider_combo = QComboBox()
    view.provider_combo.setSizePolicy(
        QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed
    )
    view.provider_combo.setMinimumWidth(0)
    mode_row.addWidget(view.provider_combo)

    view.terminal_start_btn = QPushButton("Start")
    view.terminal_start_btn.clicked.connect(dock._on_terminal_start)
    mode_row.addWidget(view.terminal_start_btn)

    view.terminal_stop_btn = QPushButton("Stop")
    view.terminal_stop_btn.clicked.connect(dock._on_terminal_stop)
    mode_row.addWidget(view.terminal_stop_btn)

    # Message scroll area
    view.scroll_area = QScrollArea()
    view.scroll_area.setWidgetResizable(True)
    view.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    view.message_container = QWidget()
    view.message_layout = QVBoxLayout(view.message_container)
    view.message_layout.setContentsMargins(2, 2, 2, 2)
    view.message_layout.setSpacing(4)
    view.message_layout.addStretch()
    view.scroll_area.setWidget(view.message_container)

    view.terminal_output = QPlainTextEdit()
    view.terminal_output.setReadOnly(True)
    view.terminal_output.setStyleSheet(
        "QPlainTextEdit {"
        f"  background-color: {bg_alt}; color: {text_primary};"
        f"  border: 1px solid {border}; border-radius: 4px;"
        "  font-family: Consolas, monospace; font-size: 12px; padding: 4px;"
        "}"
    )

    view.content_stack = QStackedWidget()
    view.content_stack.addWidget(view.scroll_area)
    view.content_stack.addWidget(view.terminal_output)
    chat_layout.addWidget(view.content_stack, stretch=1)

    # Thinking indicator — animated "Sidekick is thinking ●●●" pulser.
    # Placed between the message stack and the input row so it sits
    # immediately above whatever the user is typing, making the active
    # state immediately discoverable.
    view.thinking_indicator = ThinkingIndicator(
        parent=dock,
        theme_provider=dock._theme_provider,
        accent_color=dock._accent_color,
    )
    chat_layout.addWidget(view.thinking_indicator)

    # Inline preview of the busy-state message queue. Hidden when the
    # queue is empty; surfaces each queued steering message with its own
    # steer-to-front button.
    view.queue_panel = QueuePanel(parent=dock)
    view.queue_panel.steer_requested.connect(dock.steer_to_front)
    chat_layout.addWidget(view.queue_panel)

    # Input row
    input_row = QHBoxLayout()
    view.input_edit = QPlainTextEdit()
    install_enter_submit(view.input_edit, dock._on_send)
    view.input_edit.setMinimumHeight(60)
    view.input_edit.setMaximumHeight(150)
    view.input_edit.setSizePolicy(
        QSizePolicy.Policy.Ignored, QSizePolicy.Policy.MinimumExpanding
    )
    view.input_edit.setPlaceholderText(dock._placeholder_text)
    view.input_edit.setStyleSheet(
        "QPlainTextEdit {"
        f"  background-color: {bg_alt}; color: {text_primary};"
        f"  border: 1px solid {border}; border-radius: 4px;"
        "  font-size: 12px; padding: 4px;"
        "}"
    )
    chat_layout.addWidget(view.input_edit)

    # Tools on the far left
    view.tools_btn.setFixedWidth(50)
    view.tools_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
    view.tools_btn.setStyleSheet(
        "QPushButton {"
        f"  background-color: {bg_alt}; color: {text_primary};"
        "  border-radius: 4px; padding: 4px;"
        "}"
        f"QPushButton:hover {{ background-color: {border}; }}"
    )
    input_row.addWidget(view.tools_btn)

    view.upload_btn = _make_icon_button(
        "+", "Upload file", bg_alt, text_primary, border, dock._on_upload
    )
    input_row.addWidget(view.upload_btn)

    view.screenshot_btn = _make_icon_button(
        "⛶", "Capture screenshot", bg_alt, text_primary, border, dock._on_screenshot
    )
    input_row.addWidget(view.screenshot_btn)

    view.mic_btn = _make_icon_button(
        "\U0001f3a4",
        "Voice input (Ctrl+Shift+V)",
        bg_alt,
        text_primary,
        border,
        dock._on_mic_toggle,
    )
    input_row.addWidget(view.mic_btn)

    input_row.addStretch()

    view.agent_mode_combo = QComboBox()
    view.agent_mode_combo.setSizePolicy(
        QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding
    )
    view.agent_mode_combo.addItem("Agent", "agent")
    view.agent_mode_combo.addItem("Plan", "plan")
    view.agent_mode_combo.addItem("Ask", "ask")
    input_row.addWidget(view.agent_mode_combo)

    # Send, Steer, Stop on the right side
    send_style = (
        "QPushButton {"
        f"  background-color: {dock._accent_color}; color: black;"
        "  border-radius: 4px; font-weight: bold; padding: 4px;"
        "}"
        f"QPushButton:hover {{ background-color: {button_hover}; }}"
        "QPushButton:disabled { background-color: #555; color: #888; }"
    )
    view.send_btn = _make_action_button(
        "Send", "Send message", 55, send_style, dock._on_send
    )
    input_row.addWidget(view.send_btn)

    view.steer_btn = _make_action_button(
        "Steer", "Queue message", 50, send_style, dock._on_steer
    )
    input_row.addWidget(view.steer_btn)

    view.stop_agent_btn = _make_action_button(
        "Stop", "Stop response", 50, send_style, dock._on_stop_agent
    )
    input_row.addWidget(view.stop_agent_btn)

    chat_layout.addLayout(input_row)
    chat_layout.addLayout(mode_row)
    dock.setWidget(container)

    # Dock widget styling
    dock.setStyleSheet(
        f"QDockWidget {{ background-color: {bg_primary}; color: {text_primary}; }}"
        "QDockWidget::title {"
        f"  background-color: {dock._accent_color}; color: black;"
        "  padding: 6px; font-weight: bold;"
        "}"
    )
    view.scroll_area.setStyleSheet(
        f"QScrollArea {{ background-color: {bg_primary}; border: none; }}"
    )
    view.message_container.setStyleSheet(f"background-color: {bg_primary};")

    # Keyboard shortcut for voice input
    shortcut = QShortcut(QKeySequence("Ctrl+Shift+V"), dock)
    shortcut.activated.connect(dock._on_mic_toggle)

    # Wire voice manager callbacks
    dock._voice_manager.connect_transcription(dock._on_voice_transcription)
    dock._voice_manager.connect_error(dock._on_voice_error)
    return view


def _make_icon_button(
    text: str,
    tooltip: str,
    bg: str,
    fg: str,
    border: str,
    on_clicked: Any,
) -> QPushButton:
    """DRY helper for the small fixed-width icon buttons (upload, screenshot, mic)."""
    btn = QPushButton(text)
    btn.setToolTip(tooltip)
    btn.setFixedWidth(28)
    btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
    btn.setStyleSheet(
        "QPushButton {"
        f"  background-color: {bg}; color: {fg};"
        "  border-radius: 4px; padding: 4px;"
        "}"
        f"QPushButton:hover {{ background-color: {border}; }}"
    )
    btn.clicked.connect(on_clicked)
    return btn


def _make_action_button(
    text: str,
    tooltip: str,
    width: int,
    style: str,
    on_clicked: Any,
) -> QPushButton:
    """DRY helper for the right-aligned action buttons (Send/Steer/Stop)."""
    btn = QPushButton(text)
    btn.setToolTip(tooltip)
    btn.setFixedWidth(width)
    btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
    btn.setStyleSheet(style)
    btn.clicked.connect(on_clicked)
    return btn
