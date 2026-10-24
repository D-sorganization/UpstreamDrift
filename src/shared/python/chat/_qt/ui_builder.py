"""Compatibility declarations for the retired chat UI builder implementation.

Public imports resolve through the pinned Tools alias so both supported module
spellings share the provider's implementation. The remaining local view and
button declarations are retained until a follow-up retirement slice.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QComboBox,
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
from .history_sidebar import HistorySidebar
from .queue_panel import QueuePanel


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
