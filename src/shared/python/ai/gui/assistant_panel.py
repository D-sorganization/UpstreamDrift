"""Backward-compatibility shim for the AI Assistant Panel.

The implementation has been split into sub-modules under
``src.shared.python.ai.gui.assistant``:

- ``composer``    — ChatInput and input-area helpers
- ``transcript``  — MessageWidget and message-area builder
- ``streaming``   — StreamWorker
- ``panel``       — AIAssistantPanel orchestration

This module re-exports all public names from those sub-modules so
existing ``from src.shared.python.ai.gui.assistant_panel import X``
call-sites continue to work without modification.
"""

from __future__ import annotations

# Re-export all public names from their canonical locations.
from src.shared.python.ai.gui.assistant.composer import ChatInput
from src.shared.python.ai.gui.assistant.panel import (
    AIAssistantPanel,
    _rust_ollama_endpoint_paths,
)
from src.shared.python.ai.gui.assistant.streaming import StreamWorker
from src.shared.python.ai.gui.assistant.transcript import MessageWidget

__all__ = [
    "AIAssistantPanel",
    "ChatInput",
    "MessageWidget",
    "StreamWorker",
    "_rust_ollama_endpoint_paths",
]
