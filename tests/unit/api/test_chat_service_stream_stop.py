"""Regression tests for chat stream worker cancellation (issue #6981).

The async ``ChatService.stream_response`` spawns a daemon worker thread
(``_stream_to_queue``) that pulls chunks from the adapter into a queue.
Before the fix, on client disconnect the async consumer stopped reading
but the worker kept running -- pulling from the adapter, taking the lock
to persist messages for an abandoned session, and leaking the daemon
thread. These tests verify a ``threading.Event`` stop flag is set on
disconnect and the worker observes it and exits promptly.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


def _make_chat_service(tmp_path) -> Any:
    """Return a ChatService with a mocked adapter and temp persist dir."""
    with patch("src.api.services.chat_service.ChatService._load_adapter"):
        from src.api.services.chat_service import ChatService

        svc = ChatService()
        svc.PERSIST_DIR = tmp_path / "chat_sessions"
        svc._adapter = MagicMock()
        return svc


class _SlowAdapter:
    """Adapter whose stream yields chunks slowly and forever.

    Records whether the generator was abandoned mid-stream (i.e. the
    consumer stopped iterating) via ``chunks_yielded``. Each yielded
    chunk waits on ``chunk_gate`` so the test can control timing and
    detect that the worker stopped pulling after the stop flag is set.
    """

    def __init__(self) -> None:
        self.chunks_yielded = 0
        self.stopped_iterating = threading.Event()

    def stream_response(self, _message, _ctx, _tools):  # noqa: ANN001, ANN201
        from src.shared.python.ai.types import AgentChunk

        try:
            while True:
                self.chunks_yielded += 1
                yield AgentChunk(content="tok ")
                time.sleep(0.02)
        finally:
            # Reached when the worker breaks out / generator is closed.
            self.stopped_iterating.set()


@pytest.mark.unit
def test_stream_stop_event_halts_worker_on_disconnect(tmp_path) -> None:
    """On consumer disconnect the worker sees the stop flag and exits.

    Simulates a client disconnect by closing the async generator early
    (``aclose``), then asserts: (1) no ``_stream_to_queue`` worker thread
    is left alive, and (2) the adapter stream was actually abandoned.
    """
    svc = _make_chat_service(tmp_path)
    adapter = _SlowAdapter()
    svc._adapter = adapter

    ctx = svc.get_or_create_session(None)
    svc.add_user_message(ctx.session_id, "stream forever please")

    threads_before = {t.ident for t in threading.enumerate()}

    async def _drive() -> None:
        gen = svc.stream_response(ctx.session_id)
        # Consume a couple of chunks, then disconnect (stop reading).
        await gen.__anext__()
        await gen.__anext__()
        # Client disconnect: closing the generator must propagate
        # GeneratorExit into stream_response and cancel the worker.
        await gen.aclose()

    asyncio.run(_drive())

    # The adapter generator must have been abandoned (its finally ran).
    assert adapter.stopped_iterating.wait(timeout=5.0), (
        "adapter stream was never cancelled -- worker kept pulling"
    )

    # No leaked worker thread: the daemon worker must have exited.
    deadline = time.monotonic() + 5.0
    leaked = []
    while time.monotonic() < deadline:
        leaked = [
            t
            for t in threading.enumerate()
            if t.ident not in threads_before and t.is_alive()
        ]
        if not leaked:
            break
        time.sleep(0.05)
    assert not leaked, f"leaked worker thread(s) after disconnect: {leaked}"
