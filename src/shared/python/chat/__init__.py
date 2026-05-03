"""Shared AI chat widget and contract models.

Provides a portable ChatDockWidget (QDockWidget + QWebSocket) that connects
to any FastAPI-based chat WebSocket endpoint, plus Pydantic contract models
for the chat protocol.

Usage::

    from chat import ChatDockWidget

    dock = ChatDockWidget(
        app_context="gasification",
        app_name="integrated_process_simulator",
        parent=main_window,
    )
    main_window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
"""

try:
    from .models import (
        ChatChunkResponse,
        ChatHistoryResponse,
        ChatMessageRequest,
        ChatSessionInfo,
    )

    _PYDANTIC_AVAILABLE = True
except ImportError:
    _PYDANTIC_AVAILABLE = False
    ChatChunkResponse = None  # type: ignore[assignment, misc]
    ChatHistoryResponse = None  # type: ignore[assignment, misc]
    ChatMessageRequest = None  # type: ignore[assignment, misc]
    ChatSessionInfo = None  # type: ignore[assignment, misc]

_PYQT6_AVAILABLE = None


def __getattr__(name: str):
    if name in {"ChatDockWidget", "ChatMessageBubble"}:
        from . import chat_dock_widget

        return getattr(chat_dock_widget, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ChatDockWidget",
    "ChatMessageBubble",
    "ChatMessageRequest",
    "ChatChunkResponse",
    "ChatSessionInfo",
    "ChatHistoryResponse",
]
