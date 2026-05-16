"""AI assistant sub-package.

Public re-exports:

    from src.shared.python.ai.gui.assistant import (
        AIAssistantPanel,
        ChatInput,
        MessageWidget,
        StreamWorker,
    )
"""

from src.shared.python.ai.gui.assistant.composer import ChatInput
from src.shared.python.ai.gui.assistant.panel import AIAssistantPanel
from src.shared.python.ai.gui.assistant.streaming import StreamWorker
from src.shared.python.ai.gui.assistant.transcript import MessageWidget

__all__ = [
    "AIAssistantPanel",
    "ChatInput",
    "MessageWidget",
    "StreamWorker",
]
