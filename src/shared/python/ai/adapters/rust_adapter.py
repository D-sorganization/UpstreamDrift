from __future__ import annotations

import logging
import queue
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.shared.python.ai.types import ChatModelInfo

from src.shared.python.ai.adapters.base import BaseAgentAdapter, ToolDeclaration
from src.shared.python.ai.types import (
    AgentChunk,
    AgentResponse,
    ConversationContext,
    ProviderCapabilities,
    ProviderCapability,
)
from src.shared.python.contracts import precondition

logger = logging.getLogger(__name__)

_WHEEL_MISSING_HINT = (
    "ai_backend Rust extension is not installed. "
    "Build it from the repo root with: "
    "`cd rust_core/ai_backend && maturin develop --features python`."
)


def _make_rust_stream_worker_class() -> type | None:
    """Return _RustStreamWorker(QThread), or None when PyQt6 is absent.

    The class is constructed lazily at call time so this module can be imported
    in headless environments that lack a working PyQt6 installation.
    """
    try:
        from PyQt6.QtCore import QThread, pyqtSignal
    except ImportError:
        return None

    class _RustStreamWorker(QThread):
        """Background worker that drains the Rust SSE stream off the main thread.

        Moves ``engine.stream_response(prompt)`` — a blocking Tokio ``block_on``
        call — onto a dedicated ``QThread`` so the Qt event loop is never stalled
        (Tools #2752).

        Signals:
            chunk_received: Emitted for each delta string.
            stream_finished: Emitted once after the last delta.
            stream_error: Emitted with an error message on exception.
        """

        chunk_received: pyqtSignal = pyqtSignal(object)
        stream_finished: pyqtSignal = pyqtSignal()
        stream_error: pyqtSignal = pyqtSignal(str)

        def __init__(self, engine: Any, full_prompt: str) -> None:
            super().__init__()
            self._engine = engine
            self._full_prompt = full_prompt
            self._stopped = False

        def stop(self) -> None:
            """Request the worker to stop emitting after the current chunk."""
            self._stopped = True

        def run(self) -> None:  # noqa: D401 — Qt override
            """Invoke the blocking Rust call and emit per-chunk signals."""
            try:
                deltas = self._engine.stream_response(self._full_prompt)
                for delta in deltas:
                    if self._stopped:
                        break
                    self.chunk_received.emit(delta)
                self.stream_finished.emit()
            except Exception as exc:  # noqa: BLE001
                self.stream_error.emit(str(exc))

    return _RustStreamWorker


class RustAgentAdapter(BaseAgentAdapter):
    """Adapter that delegates to the high-performance Rust AI backend.

    Follows the Law of Demeter by encapsulating the ``ai_backend`` extension
    inside standard adapter methods. The wheel is built per-crate via
    ``maturin develop`` from ``rust_core/ai_backend/`` (see the crate's
    ``pyproject.toml``); if the wheel isn't installed in the active
    environment we raise a clear ``ImportError`` rather than failing with a
    bare ``ModuleNotFoundError`` later in the call chain.

    Threading model
    ---------------
    ``stream_response`` moves the blocking Rust call onto a ``_RustStreamWorker``
    (a ``QThread`` subclass) so the Qt main thread's event loop is never stalled.
    When PyQt6 is unavailable the implementation falls back to a plain
    ``threading.Thread`` with the same queue-based handoff, preserving correct
    behaviour in headless / non-GUI environments.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        db_path: str = "./memory.db",
        *,
        chat_path: str | None = None,
        embed_path: str | None = None,
        embedding_model: str | None = None,
        use_local_embeddings: bool = False,
    ) -> None:
        """Initialize the Rust Agent Adapter.

        Args:
            api_key: The API key for the LLM.
            base_url: The base URL for the LLM endpoint
                (e.g. ``https://api.openai.com/v1``).
            model: The chat model name.
            db_path: Path to the local vector database.
            chat_path: Path suffix appended to ``base_url`` for chat
                completions. Defaults to ``/chat/completions``.
            embed_path: Path suffix for embeddings. Defaults to
                ``/embeddings``.
            embedding_model: Embedding model name. Defaults to
                ``text-embedding-3-small``.
            use_local_embeddings: When ``True``, embed via a local ONNX
                ``all-MiniLM-L6-v2`` model instead of the HTTP endpoint.
                Requires ``ai_backend`` to be built with
                ``--features python,local-embeddings``. The model is
                cached at ``$UPSTREAM_DRIFT_MODEL_CACHE`` or
                ``~/.cache/upstream-drift/models/`` and downloaded on
                first use.
        """
        try:
            import ai_backend
        except ImportError as exc:
            logger.warning(
                "rust_adapter: ai_backend wheel not available; using pure-Python path. "
                "See docs/development/rust_distribution.md"
            )
            raise ImportError(_WHEEL_MISSING_HINT) from exc

        self.config = ai_backend.AIConfig(
            api_key,
            base_url,
            model,
            db_path,
            chat_path,
            embed_path,
            embedding_model,
        )
        self.engine = ai_backend.AIEngine(self.config)
        self.memory = ai_backend.MemoryManager(db_path)
        self.memory.initialize()
        # The RagPipeline signature changed when local-embeddings landed
        # (#5227): older wheels won't accept the third positional flag.
        # Fall back to the old call when the new path raises a TypeError so
        # mixed-version environments don't break.
        try:
            self.rag = ai_backend.RagPipeline(
                self.memory, self.config, use_local_embeddings
            )
        except TypeError:
            if use_local_embeddings:
                raise RuntimeError(
                    "Installed ai_backend wheel does not support "
                    "use_local_embeddings; rebuild with "
                    "`maturin develop --features python,local-embeddings`."
                ) from None
            self.rag = ai_backend.RagPipeline(self.memory, self.config)

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def stream_response(
        self,
        prompt: str,
        context: ConversationContext,
        tools: list[Any],
    ) -> Iterator[AgentChunk]:
        """Stream the response using the Rust backend off the Qt main thread.

        Delegates the blocking ``engine.stream_response`` call to a
        ``_RustStreamWorker`` (``QThread``) when PyQt6 is available, keeping the
        Qt event loop responsive throughout (fixes Tools #2752 UI freeze).
        Falls back to a daemon ``threading.Thread`` in headless environments.

        Cancel support: ``self._active_worker.stop()`` can be called from the
        UI to halt emission after the current chunk.
        """
        try:
            full_prompt = (
                "\n".join([m.content for m in context.messages]) + f"\n{prompt}"
            )
            yield from self._stream_via_qthread(full_prompt)
        except Exception as e:
            logger.exception("Rust backend error")
            yield AgentChunk(content=f"Error: {e}", is_final=True)

    def _stream_via_qthread(self, full_prompt: str) -> Iterator[AgentChunk]:
        """Dispatch the blocking call off-thread and yield AgentChunk objects."""
        result_queue: queue.Queue[Any] = queue.Queue()
        worker_cls = _make_rust_stream_worker_class()
        if worker_cls is not None:
            yield from self._stream_with_worker(worker_cls, full_prompt, result_queue)
            return
        yield from self._stream_with_thread(full_prompt, result_queue)

    def _stream_with_worker(
        self,
        worker_cls: type,
        full_prompt: str,
        result_queue: queue.Queue[Any],
    ) -> Iterator[AgentChunk]:
        """Stream via _RustStreamWorker(QThread); spin a QEventLoop until done."""
        from PyQt6.QtCore import QEventLoop

        chunk_buffer: list[str] = []
        finished_flag: list[bool] = [False]
        error_flag: list[str | None] = [None]

        worker = worker_cls(self.engine, full_prompt)
        self._active_worker = worker

        worker.chunk_received.connect(chunk_buffer.append)
        worker.stream_finished.connect(lambda: finished_flag.__setitem__(0, True))
        worker.stream_error.connect(lambda msg: error_flag.__setitem__(0, msg))

        loop = QEventLoop()
        worker.stream_finished.connect(loop.quit)
        worker.stream_error.connect(lambda _: loop.quit())
        worker.start()

        # Spin the event loop until the worker is done; keeps the Qt UI
        # responsive (repaints, button clicks, etc.) while Rust blocks.
        loop.exec()

        if error_flag[0] is not None:
            try:
                response = self.engine.generate_response(full_prompt)
                yield AgentChunk(content=response, is_final=True)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Rust fallback error")
                yield AgentChunk(content=f"Error: {exc}", is_final=True)
            return

        deltas = chunk_buffer
        if not deltas:
            yield AgentChunk(content="", is_final=True)
            return

        last_idx = len(deltas) - 1
        for idx, delta in enumerate(deltas):
            yield AgentChunk(content=delta, is_final=(idx == last_idx))

    def _stream_with_thread(
        self,
        full_prompt: str,
        result_queue: queue.Queue[Any],
    ) -> Iterator[AgentChunk]:
        """Stream via a plain daemon thread (headless / no PyQt6 fallback)."""
        import threading
        import time

        def _fetch() -> None:
            try:
                result_queue.put(self.engine.stream_response(full_prompt))
            except Exception as exc:  # noqa: BLE001
                result_queue.put(exc)

        t = threading.Thread(target=_fetch, daemon=True)
        t.start()
        while t.is_alive() and result_queue.empty():
            time.sleep(0.01)

        result = result_queue.get()
        if isinstance(result, Exception):
            try:
                response = self.engine.generate_response(full_prompt)
                yield AgentChunk(content=response, is_final=True)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Rust fallback error")
                yield AgentChunk(content=f"Error: {exc}", is_final=True)
            return

        deltas = result
        if not deltas:
            yield AgentChunk(content="", is_final=True)
            return

        last_idx = len(deltas) - 1
        for idx, delta in enumerate(deltas):
            yield AgentChunk(content=delta, is_final=(idx == last_idx))

    # ------------------------------------------------------------------
    # Other adapter methods
    # ------------------------------------------------------------------

    def index_codebase(self, root_path: str) -> int:
        """Trigger the Rust-based RAG pipeline indexer."""
        return int(self.rag.index_codebase(root_path))

    def retrieve_context(self, prompt: str, top_k: int = 5) -> list[str]:
        """Retrieve semantic context using the Rust vector memory."""
        return [str(item) for item in self.rag.retrieve_context(prompt, top_k)]

    @precondition(
        lambda prompt: bool(prompt.strip()), "message must not be empty or blank"
    )
    def send_message(
        self, prompt: str, context: ConversationContext, tools: list[ToolDeclaration]
    ) -> AgentResponse:
        """Send a message synchronously."""
        del tools
        # Canonical zero-usage so callers always see the same keys (issue #2763).
        canonical_usage = self._normalize_token_counts({})
        try:
            full_prompt = (
                "\n".join([m.content for m in context.messages]) + f"\n{prompt}"
            )
            response = self.engine.generate_response(full_prompt)
            return AgentResponse(content=str(response), usage=canonical_usage)
        except Exception as e:
            logger.exception("Rust backend error")
            return AgentResponse(
                content=f"Error: {e}",
                finish_reason="error",
                usage=canonical_usage,
            )

    def validate_connection(self) -> tuple[bool, str]:
        """Validate connection to the backend."""
        return True, "Rust backend initialized successfully."

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Return adapter capabilities."""
        return ProviderCapabilities(
            supported=frozenset(
                {
                    ProviderCapability.STREAMING,
                    ProviderCapability.SYSTEM_MESSAGE,
                }
            ),
            max_tokens=8192,
            model_name=str(self.config.model),
            provider_name="rust",
        )

    # ------------------------------------------------------------------ #
    # Tools issue #2871: provider catalogue + reasoning capabilities
    # ------------------------------------------------------------------ #

    _STATIC_MODELS: tuple[str, ...] = (
        "gpt-4-turbo",
        "gpt-4o",
        "gpt-3.5-turbo",
    )

    def list_models(self) -> list[ChatModelInfo]:
        """Return Rust-adapter model catalogue; configured model is always present."""
        from src.shared.python.ai.types import ChatModelInfo

        configured = str(getattr(self.config, "model", "") or "")
        models = list(self._STATIC_MODELS)
        if configured and configured not in models:
            models.insert(0, configured)
        return [
            ChatModelInfo(
                name=m,
                provider="rust",
                display_name=m,
                id=m,
            )
            for m in models
        ]

    def thinking_capabilities(self):  # type: ignore[no-untyped-def]
        """Rust adapter does not currently surface reasoning budgets."""
        from src.shared.python.chat.models import make_none_only_capabilities

        return make_none_only_capabilities(provider="rust")
