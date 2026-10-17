"""Shared AI chat widget, service base, and contract models.

Provides a portable ChatDockWidget (QDockWidget + QWebSocket) that connects
to any FastAPI-based chat WebSocket endpoint, plus Pydantic contract models
for the chat protocol, a shared ChatServiceBase for session management,
and a reusable WebSocket router factory.

Usage::

    from chat import ChatDockWidget, ChatConnectionConfig

    dock = ChatDockWidget(
        connection=ChatConnectionConfig(
            app_context="gasification",
            app_name="integrated_process_simulator",
        ),
        parent=main_window,
    )
    main_window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
"""

# mypy: disable-error-code="unused-ignore"
# The optional-pydantic import block below uses type: ignore comments that are
# redundant when mypy runs with --follow-imports=skip (CI) but required locally.

from typing import Any

from .cli_provider_availability import (
    CliProviderEntry,
    list_available_cli_providers,
)
from .qt_diagnostics import ChatQtDiagnostic, diagnose_chat_qt_runtime
from .service_base import ChatMessage, ChatServiceBase, ChatSession
from .terminal_contracts import (
    TerminalAgentEvent,
    TerminalAgentProviderInfo,
    TerminalAgentSessionInfo,
    TerminalAgentSessionRequest,
    TerminalProviderRegistry,
    TerminalRegistryError,
    TerminalShellInfo,
)
from .terminal_providers import (
    build_default_terminal_provider_registry,
    default_terminal_agent_providers,
    default_terminal_shells,
    provider_probe_commands,
    redact_terminal_command,
)
from .terminal_runtime import (
    ProcessLaunchRequest,
    TerminalProcessAdapter,
    TerminalRuntimeError,
    TerminalSessionRuntime,
)

try:
    from .models import (
        DEFAULT_RESPONSE_STYLE,
        RESPONSE_STYLE_PROMPTS,
        ChatChunkResponse,
        ChatHistoryResponse,
        ChatIndexStatusResponse,
        ChatMessageRequest,
        ChatModelInfo,
        ChatModelListResponse,
        ChatSessionInfo,
        ResponseStyle,
        style_prompt,
    )

    _PYDANTIC_AVAILABLE = True
except ImportError:
    _PYDANTIC_AVAILABLE = False
    ChatChunkResponse = None  # type: ignore[assignment, misc]
    ChatHistoryResponse = None  # type: ignore[assignment, misc]
    ChatIndexStatusResponse = None  # type: ignore[assignment, misc]
    ChatMessageRequest = None  # type: ignore[assignment, misc]
    ChatModelInfo = None  # type: ignore[assignment, misc]
    ChatModelListResponse = None  # type: ignore[assignment, misc]
    ChatSessionInfo = None  # type: ignore[assignment, misc]
    DEFAULT_RESPONSE_STYLE = "standard"
    RESPONSE_STYLE_PROMPTS = {}
    ResponseStyle = str  # type: ignore[assignment, misc]
    style_prompt = None  # type: ignore[assignment, misc]

_PYQT6_AVAILABLE = None


def __getattr__(name: str) -> Any:
    if name in {"ChatDockWidget", "ChatMessageBubble"}:
        from . import chat_dock_widget

        return getattr(chat_dock_widget, name)
    if name == "VoiceInputManager":
        from .voice_input_manager import VoiceInputManager

        return VoiceInputManager
    if name == "create_chat_router":
        from .router_factory import create_chat_router

        return create_chat_router
    if name in {
        "ChatWebSocketState",
        "DisconnectLogConfig",
        "run_chat_websocket_protocol",
    }:
        from . import websocket_protocol

        return getattr(websocket_protocol, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ChatDockWidget",
    "ChatMessageBubble",
    "VoiceInputManager",
    "ChatMessageRequest",
    "ChatChunkResponse",
    "ChatSessionInfo",
    "ChatHistoryResponse",
    "ChatModelInfo",
    "ChatModelListResponse",
    "ChatIndexStatusResponse",
    "ResponseStyle",
    "DEFAULT_RESPONSE_STYLE",
    "RESPONSE_STYLE_PROMPTS",
    "style_prompt",
    "ChatServiceBase",
    "ChatSession",
    "ChatMessage",
    "ChatWebSocketState",
    "DisconnectLogConfig",
    "create_chat_router",
    "run_chat_websocket_protocol",
    "TerminalAgentEvent",
    "TerminalAgentProviderInfo",
    "TerminalAgentSessionInfo",
    "TerminalAgentSessionRequest",
    "TerminalProviderRegistry",
    "TerminalRegistryError",
    "TerminalShellInfo",
    "build_default_terminal_provider_registry",
    "default_terminal_agent_providers",
    "default_terminal_shells",
    "provider_probe_commands",
    "redact_terminal_command",
    "ProcessLaunchRequest",
    "TerminalProcessAdapter",
    "TerminalRuntimeError",
    "TerminalSessionRuntime",
    "CliProviderEntry",
    "list_available_cli_providers",
    "ChatQtDiagnostic",
    "diagnose_chat_qt_runtime",
]
