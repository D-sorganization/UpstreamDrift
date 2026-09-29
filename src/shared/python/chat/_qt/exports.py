# ruff: noqa: E501
"""Export, condense, and token-counter helpers for ``ChatDockWidget``.

Extracted from the monolithic ``_chat_dock_widget_qt`` module so the
parent file fits in the repo's 1500-line budget. All helpers take the
chat-dock instance explicitly — no hidden state, no method chains
deeper than two levels (LOD).
"""

from __future__ import annotations

import logging
from typing import Any, Literal, cast

from PyQt6.QtWidgets import QApplication, QFileDialog

from .bubbles import ChatMessageBubble

logger = logging.getLogger(__name__)


def get_thread_markdown(dock: Any) -> str:
    """Render the visible message thread as Markdown."""
    lines: list[str] = []
    for i in range(dock._message_layout.count()):
        item = dock._message_layout.itemAt(i)
        if item:
            widget = item.widget()
            if isinstance(widget, ChatMessageBubble):
                role_str = "You" if widget._role == "user" else "AI"
                lines.append(f"**{role_str}**:\n\n{widget._content}\n")
    return "\n".join(lines)


def copy_entire_thread(dock: Any) -> None:
    """Copy the full thread markdown to the system clipboard."""
    clipboard = QApplication.clipboard()
    if clipboard is not None:
        clipboard.setText(get_thread_markdown(dock))
        dock._status_label.setText("Thread copied to clipboard")
        dock._status_label.setStyleSheet("color: #3fb950; font-size: 10px;")


def build_session_snapshot(dock: Any) -> Any:
    """Materialise the visible thread as a :class:`ChatSession`."""
    from ..service_base import ChatSession

    session = ChatSession(session_id=dock._get_shared_session_id() or "session")
    for i in range(dock._message_layout.count()):
        item = dock._message_layout.itemAt(i)
        if item is None:
            continue
        widget = item.widget()
        if isinstance(widget, ChatMessageBubble):
            session.add_message(widget._role, widget._content)
    return session


def export_thread(dock: Any, fmt: str, file_filter: str, suffix: str) -> None:
    """Run an export via the shared ``chat.export`` package."""
    from ..export import (
        ChatExportRequest,
        HtmlExporter,
        MarkdownExporter,
        TextExporter,
    )

    path, _ = QFileDialog.getSaveFileName(
        dock,
        "Export Chat Thread",
        str(dock._project_root / f"chat_export{suffix}"),
        f"{file_filter};;All Files (*)",
    )
    if not path:
        return
    session = build_session_snapshot(dock)
    if session is None or session.message_count == 0:
        dock._status_label.setText("Nothing to export")
        dock._status_label.setStyleSheet("color: #f85149; font-size: 10px;")
        return
    request = ChatExportRequest(
        session_id=session.session_id,
        format=cast("Literal['markdown', 'text', 'html']", fmt),
        output_path=path,
        include_metadata=True,
        redact_secrets=True,
    )
    try:
        if fmt == "markdown":
            result = MarkdownExporter().export(session, request)
        elif fmt == "text":
            result = TextExporter().export(session, request)
        elif fmt == "html":
            result = HtmlExporter().export(session, request)
        else:
            raise ValueError(f"Unknown export format {fmt!r}")
        dock._status_label.setText(
            f"Exported {result.message_count} messages ({result.byte_count} B)"
        )
        dock._status_label.setStyleSheet("color: #3fb950; font-size: 10px;")
    except (OSError, ValueError) as exc:
        dock._status_label.setText(f"Export error: {exc}")
        dock._status_label.setStyleSheet("color: #f85149; font-size: 10px;")


def run_condense_local(dock: Any, strategy: str) -> None:
    """Run condensation locally via the shared ``chat.condensation`` package."""
    from ..condensation import CondensationRequest, Condenser

    session = build_session_snapshot(dock)
    if session is None or session.message_count == 0:
        dock._status_label.setText("Nothing to condense")
        dock._status_label.setStyleSheet("color: #f85149; font-size: 10px;")
        return
    try:
        request = CondensationRequest(
            session_id=session.session_id,
            strategy=cast(
                "Literal['keep_recent', 'semantic_summary', 'pinned_anchor']",
                strategy,
            ),
            keep_last_n=max(1, min(10, session.message_count)),
        )
        result = Condenser().condense(session, request)
    except ValueError as exc:
        dock._status_label.setText(f"Condense error: {exc}")
        dock._status_label.setStyleSheet("color: #f85149; font-size: 10px;")
        return
    dock._status_label.setText(
        f"Condense [{strategy}]: {result.original_message_count} -> "
        f"{result.condensed_message_count} msgs, "
        f"~{result.removed_tokens_estimate} tok saved"
    )
    dock._status_label.setStyleSheet("color: #3fb950; font-size: 10px;")
    refresh_token_indicator(dock)


def refresh_token_indicator(dock: Any) -> None:
    """Recompute the token-count indicator label."""
    from ..condensation import estimate_tokens

    if not hasattr(dock, "_token_indicator"):
        return
    total = 0
    for i in range(dock._message_layout.count()):
        item = dock._message_layout.itemAt(i)
        if item is None:
            continue
        widget = item.widget()
        if isinstance(widget, ChatMessageBubble):
            total += estimate_tokens(widget._content)
    dock._token_indicator.setText(f"{total} tok")
    if total > dock._auto_condense_threshold:
        dock._token_indicator.setStyleSheet("color: #f85149; font-size: 10px;")
    else:
        dock._token_indicator.setStyleSheet("color: #8b949e; font-size: 10px;")
