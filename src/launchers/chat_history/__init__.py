"""Launcher-side chat history + persistent memory management UI.

Surfaces the Sidekick (Tools repo) conversation management API and the
``user_memory.json`` archival document inside the UpstreamDrift launcher's
chat panel.

Implements UpstreamDrift #5621 (re-file of phantom-closed #5370 / #5371 /
#5372). Depends on Tools #2879 (already merged) for the backend
``conversation`` service: ``list / search / archive / unarchive / delete /
export / load_as_context`` — and, optionally, Tools #2736 for memory
condensation (``condense_to_memory``); the adapter falls back to a stub
when the condenser is not yet available.
"""

from src.launchers.chat_history.chat_history_service import (
    HistoryServiceAdapter,
)
from src.launchers.chat_history.history_pane import HistorySidebarPane
from src.launchers.chat_history.memory_panel import MemoryPanel

__all__ = [
    "HistoryServiceAdapter",
    "HistorySidebarPane",
    "MemoryPanel",
]
