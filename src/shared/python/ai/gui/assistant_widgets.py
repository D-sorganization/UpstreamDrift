"""Small reusable widgets for the shared AI assistant panel.

This module now re-exports classes from the decomposed `assistant` sub-package
to maintain backward compatibility.
"""

from __future__ import annotations

from src.shared.python.ai.gui.assistant.composer import ChatInput
from src.shared.python.ai.gui.assistant.streaming import StreamWorker
from src.shared.python.ai.gui.assistant.transcript import MessageWidget

__all__ = [
    "ChatInput",
    "MessageWidget",
    "StreamWorker",
]
