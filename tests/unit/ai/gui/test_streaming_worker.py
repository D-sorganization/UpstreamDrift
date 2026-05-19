"""Tests for StreamWorker runtime behavior and error handling."""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest

try:
    import PyQt6.QtCore  # noqa: F401
except (ImportError, OSError) as _exc:
    pytest.skip(f"PyQt6 not loadable: {_exc}", allow_module_level=True)

from src.shared.python.ai.gui.assistant.streaming import StreamWorker
from src.shared.python.ai.exceptions import AIProviderError
from src.shared.python.ai.types import ConversationContext, AgentChunk

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_stream_worker_success() -> None:
    """StreamWorker emits chunk_received signals for each response chunk and finished at the end."""
    adapter = MagicMock()
    context = ConversationContext()
    tools = []

    # Mock stream_response to yield some chunks
    chunks = [
        AgentChunk(content="Hello"),
        AgentChunk(content=" World"),
    ]
    adapter.stream_response.return_value = chunks

    worker = StreamWorker(adapter, "test message", context, tools)

    received_chunks = []
    worker._finished_called = False
    error_called = None

    worker.chunk_received.connect(received_chunks.append)
    worker.finished.connect(lambda: setattr(worker, "_finished_called", True))

    def on_error(msg: str) -> None:
        nonlocal error_called
        error_called = msg

    worker.error.connect(on_error)

    worker.run()

    assert received_chunks == ["Hello", " World"]
    assert worker._finished_called is True
    assert error_called is None
    adapter.stream_response.assert_called_once_with("test message", context, tools)


def test_stream_worker_handles_ai_provider_error() -> None:
    """StreamWorker catches AIProviderError, emits error and finished signals."""
    adapter = MagicMock()
    context = ConversationContext()
    tools = []

    def failing_generator(*args, **kwargs):
        yield AgentChunk(content="Start")
        raise AIProviderError("Ollama failed to respond", provider="ollama")

    adapter.stream_response.side_effect = failing_generator

    worker = StreamWorker(adapter, "test message", context, tools)

    received_chunks = []
    error_called = None
    worker._finished_called = False

    worker.chunk_received.connect(received_chunks.append)
    worker.finished.connect(lambda: setattr(worker, "_finished_called", True))

    def on_error(msg: str) -> None:
        nonlocal error_called
        error_called = msg

    worker.error.connect(on_error)

    worker.run()

    assert received_chunks == ["Start"]
    assert worker._finished_called is True
    assert error_called is not None
    assert "Ollama failed to respond" in error_called
