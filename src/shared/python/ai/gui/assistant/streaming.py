"""Streaming worker for AI responses.

Provides StreamWorker, a QThread subclass that streams chunks from the
selected AI adapter and emits chunk_received / finished / error signals.
"""

from __future__ import annotations

from src.shared.python.ai.gui.assistant._guards import require_not_none

from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import QThread, pyqtSignal

from src.shared.python.logging_pkg.logging_config import get_logger

if TYPE_CHECKING:
    from src.shared.python.ai.adapters.base import BaseAgentAdapter

from src.shared.python.ai.types import ConversationContext

logger = get_logger(__name__)


class StreamWorker(QThread):
    """Worker thread for streaming AI responses."""

    chunk_received = pyqtSignal(str)  # Emits content chunk
    finished = pyqtSignal()  # Emits when complete
    error = pyqtSignal(str)  # Emits error message

    def __init__(
        self,
        adapter: BaseAgentAdapter,
        message: str,
        context: ConversationContext,
        tools: list[Any],
    ) -> None:
        """Initialize stream worker.

        Args:
            adapter: AI adapter to use.
            message: User message.
            context: Conversation context.
            tools: Available tools.
        """
        require_not_none(adapter, "adapter")
        super().__init__()
        self._adapter = adapter
        self._message = message
        self._context = context
        self._tools = tools

    def run(self) -> None:
        """Execute streaming in background thread."""
        try:
            for chunk in self._adapter.stream_response(
                self._message,
                self._context,
                self._tools,
            ):
                if chunk.content:
                    self.chunk_received.emit(chunk.content)
        except (RuntimeError, ValueError, OSError) as e:
            logger.exception("Streaming error")
            self.error.emit(str(e))
        finally:
            self.finished.emit()
