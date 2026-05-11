from __future__ import annotations

from typing import TYPE_CHECKING, Any
from collections.abc import Iterator
import logging

from src.shared.python.ai.adapters.base import BaseAgentAdapter
from src.shared.python.ai.types import AgentChunk, ConversationContext

logger = logging.getLogger(__name__)


class RustAgentAdapter(BaseAgentAdapter):
    """Adapter that delegates to the high-performance Rust AI backend.

    This follows the Law of Demeter by encapsulating the ai_backend
    library inside standard adapter methods.
    """

    def __init__(
        self, api_key: str, base_url: str, model: str, db_path: str = "./memory.db"
    ) -> None:
        """Initialize the Rust Agent Adapter.

        Args:
            api_key: The API key for the LLM.
            base_url: The base URL for the LLM endpoint.
            model: The name of the model to use.
            db_path: Path to the local vector database.
        """
        import ai_backend

        self.config = ai_backend.AIConfig(api_key, base_url, model, db_path)
        self.engine = ai_backend.AIEngine(self.config)
        self.memory = ai_backend.MemoryManager(db_path)
        self.rag = ai_backend.RagPipeline(self.memory)

    def stream_response(
        self,
        prompt: str,
        context: ConversationContext,
        tools: list[Any],
    ) -> Iterator[AgentChunk]:
        """Streams response using the Rust backend.

        Note: The Rust backend currently uses a blocking generate_response
        for simplicity, but we yield it as a chunk to fulfill the interface contract.
        """
        try:
            # Build full context
            full_prompt = (
                "\n".join([m.content for m in context.messages]) + f"\n{prompt}"
            )

            # Delegate to Rust core
            response = self.engine.generate_response(full_prompt)
            yield AgentChunk(content=response, is_final=True)
        except Exception as e:
            logger.error(f"Rust backend error: {e}")
            yield AgentChunk(content=f"Error: {e}", is_final=True)

    def index_codebase(self, root_path: str) -> int:
        """Triggers the Rust-based high-performance RAG pipeline."""
        return self.rag.index_codebase(root_path)

    def retrieve_context(self, prompt: str, top_k: int = 5) -> list[str]:
        """Retrieves semantic context using the Rust vector memory."""
        return self.rag.retrieve_context(prompt, top_k)
