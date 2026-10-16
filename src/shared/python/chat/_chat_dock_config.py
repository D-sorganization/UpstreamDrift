"""Configuration objects for :class:`chat.ChatDockWidget`.

Split out of ``_chat_dock_widget_qt`` (Tools #4896) so the constructor's
parameter groups stay importable without PyQt6 and the widget module stays
under the 1,200-line file budget. ``_chat_dock_widget_qt`` and ``chat``
re-export every name, so existing imports keep working.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._theme_protocol import ThemeProviderProtocol
from ._workspace_protocol import WorkspaceContextProtocol
from .terminal_contracts import TerminalProviderRegistry

__all__ = [
    "ChatConnectionConfig",
    "ChatIntegrationHooks",
    "ChatPresentationConfig",
]


@dataclass
class ChatConnectionConfig:
    """Connection + session-identity settings for :class:`ChatDockWidget`.

    Groups the WebSocket connection parameters (``server_url``,
    ``session_id``, ``ws_path_template``) with the identity/session
    parameters (``app_context``, ``app_name``, ``project_root``) — the
    latter travel with the former because they are baked into every
    outgoing WS payload and the session-file path. Tools issue #4896.
    """

    server_url: str | None = None
    session_id: str | None = None
    ws_path_template: str = "/api/ws/chat/{session_id}"
    app_context: str = "unknown"
    app_name: str = "shared_chat"
    project_root: str | Path | None = None


@dataclass
class ChatPresentationConfig:
    """Presentation/theming settings for :class:`ChatDockWidget`.

    Tools issue #4896.
    """

    placeholder_text: str = "Ask a question..."
    accent_color: str = "#FF8800"
    theme_provider: ThemeProviderProtocol | None = None
    auto_index_on_open: bool = False


@dataclass
class ChatIntegrationHooks:
    """Dependency-injection hooks for host-application integrations.

    Every field is an optional collaborator the widget falls back to a
    default implementation for when unset (terminal registry, session
    manager, memory manager factory) or simply leaves inert (workspace
    provider, plot request sink). Tools issue #4896.
    """

    terminal_registry: TerminalProviderRegistry | None = None
    workspace_provider: WorkspaceContextProtocol | None = None
    plot_request_sink: Callable[[Any], None] | None = None
    session_manager: Any | None = None
    memory_manager_factory: Callable[[], Any] | None = None
