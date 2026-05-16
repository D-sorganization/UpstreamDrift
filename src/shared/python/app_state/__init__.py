"""app_state — Application State Tracking and Diagnostic History.

Public API::

    from src.shared.python.app_state import (
        StateLogger,
        get_state_logger,
        DiagnosticEngine,
        DiagnosticResult,
        HistoryStore,
        AppEvent,
        agent_context,
    )

Layers (strict LOD):

- :mod:`._events`         — ``AppEvent`` dataclass
- :mod:`._history_store`  — ``HistoryStore`` (thread-safe ring-buffer)
- :mod:`._state_logger`   — ``StateLogger`` singleton + ``get_state_logger``
- :mod:`._diagnostic`     — ``DiagnosticEngine`` + ``DiagnosticResult``
- :mod:`._agent_context`  — ``agent_context()`` serialiser for AI agents
- :mod:`.gui`             — optional Qt widgets (headless-safe)
"""

from __future__ import annotations

from src.shared.python.app_state._agent_context import agent_context
from src.shared.python.app_state._diagnostic import DiagnosticEngine, DiagnosticResult
from src.shared.python.app_state._events import AppEvent
from src.shared.python.app_state._history_store import HistoryStore
from src.shared.python.app_state._state_logger import StateLogger, get_state_logger

__all__ = [
    "AppEvent",
    "DiagnosticEngine",
    "DiagnosticResult",
    "HistoryStore",
    "StateLogger",
    "agent_context",
    "get_state_logger",
]
