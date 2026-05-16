"""AI Assistant Panel — backward-compatible re-export shim.

The implementation has been split into focused sub-modules under
``src.shared.python.ai.gui.assistant``:

- ``composer``   — ChatInput widget and input-area builder
- ``transcript`` — MessageWidget and message-area builder
- ``streaming``  — StreamWorker (QThread)
- ``panel``      — AIAssistantPanel orchestration widget

Import directly from the sub-package for new code:

    from src.shared.python.ai.gui.assistant import AIAssistantPanel
"""

from src.shared.python.ai.gui.assistant import (  # noqa: F401
    AIAssistantPanel,
    ChatInput,
    MessageWidget,
    StreamWorker,
)

__all__ = [
    "AIAssistantPanel",
    "ChatInput",
    "MessageWidget",
    "StreamWorker",
]
