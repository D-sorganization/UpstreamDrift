from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from src.shared.python.ai.adapters.base import BaseAgentAdapter
from src.shared.python.ai.types import AgentChunk, ConversationContext

logger = logging.getLogger(__name__)

_WHEEL_MISSING_HINT = (
    "ai_backend Rust extension is not installed. "
    "Build it from the repo root with: "
    "`cd rust_core/ai_backend && maturin develop --features python`."
)


class RustAgentAdapter(BaseAgentAdapter):
    """Adapter that delegates to the high-performance Rust AI backend.

    Follows the Law of Demeter by encapsulating the ``ai_backend`` extension
    inside standard adapter methods. The wheel is built per-crate via
    ``maturin develop`` from ``rust_core/ai_backend/`` (see the crate's
    ``pyproject.toml``); if the wheel isn't installed in the active
    environment we raise a clear ``ImportError`` rather than failing with a
    bare ``ModuleNotFoundError`` later in the call chain.
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
        except ImportError as exc:  # pragma: no cover - environment-specific
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

    def stream_response(
        self,
        prompt: str,
        context: ConversationContext,
        tools: list[Any],
    ) -> Iterator[AgentChunk]:
        """Stream the response using the Rust backend.

        The Rust backend exposes ``stream_response(prompt) -> list[str]`` that
        eagerly drains the SSE stream and returns the ordered delta list.
        We re-emit each delta as an ``AgentChunk`` so callers that already
        iterate this iterator-of-chunks contract keep working. Truly-
        incremental streaming across the PyO3 boundary is a follow-up.
        """
        try:
            full_prompt = (
                "\n".join([m.content for m in context.messages]) + f"\n{prompt}"
            )

            try:
                deltas = self.engine.stream_response(full_prompt)
            except Exception:
                # Fall back to the blocking single-shot path if streaming
                # fails (e.g. provider does not support stream=true).
                response = self.engine.generate_response(full_prompt)
                yield AgentChunk(content=response, is_final=True)
                return

            if not deltas:
                yield AgentChunk(content="", is_final=True)
                return

            last_idx = len(deltas) - 1
            for idx, delta in enumerate(deltas):
                yield AgentChunk(content=delta, is_final=(idx == last_idx))
        except Exception as e:
            logger.exception("Rust backend error")
            yield AgentChunk(content=f"Error: {e}", is_final=True)

    def index_codebase(self, root_path: str) -> int:
        """Trigger the Rust-based RAG pipeline indexer."""
        return self.rag.index_codebase(root_path)

    def retrieve_context(self, prompt: str, top_k: int = 5) -> list[str]:
        """Retrieve semantic context using the Rust vector memory."""
        return self.rag.retrieve_context(prompt, top_k)
