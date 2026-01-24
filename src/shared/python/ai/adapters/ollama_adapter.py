"""Ollama adapter for local LLM inference.

This adapter enables FREE, 100% local AI assistance with no API keys
or external services required. It connects to a locally running Ollama
instance.

Requirements:
    - Ollama installed (https://ollama.ai)
    - Recommended models: llama3.1:8b, mistral, codellama
    - Minimum RAM: 8GB (16GB+ recommended for larger models)

Example:
    >>> from shared.python.ai.adapters.ollama_adapter import OllamaAdapter
    >>> adapter = OllamaAdapter()  # Uses default localhost:11434
    >>> success, message = adapter.validate_connection()
    >>> if success:
    ...     response = adapter.send_message("Hello", context, tools)
"""

from __future__ import annotations

import json
from src.shared.python.logging_config import get_logger
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

from src.shared.python.ai.adapters.base import BaseAgentAdapter, ToolDeclaration
from src.shared.python.ai.exceptions import (
    AIConnectionError,
    AIProviderError,
    AITimeoutError,
)
from src.shared.python.ai.types import (
    AgentChunk,
    AgentResponse,
    ConversationContext,
    ProviderCapabilities,
    ProviderCapability,
    ToolCall,
)

logger = get_logger(__name__)

# Default Ollama configuration
OLLAMA_DEFAULT_HOST = "http://localhost:11434"
OLLAMA_DEFAULT_MODEL = "llama3.1:8b"
OLLAMA_DEFAULT_TIMEOUT = 120.0  # [s] - Local models can be slow


class OllamaAdapter(BaseAgentAdapter):
    """Adapter for local Ollama LLM inference.

    This adapter enables FREE, 100% local AI assistance with complete
    privacy - no data ever leaves the user's machine.

    Supported Features:
        - Chat completion with conversation history
        - Streaming responses
        - Multiple model support
        - Tool calling (model-dependent)

    Attributes:
        host: Ollama server URL.
        model: Model name to use.
        timeout: Request timeout [s].

    Example:
        >>> adapter = OllamaAdapter(model="llama3.1:8b")
        >>> if adapter.validate_connection()[0]:
        ...     response = adapter.send_message(
        ...         "Help me analyze a golf swing",
        ...         context,
        ...         tools
        ...     )
    """

    def __init__(
        self,
        host: str = OLLAMA_DEFAULT_HOST,
        model: str = OLLAMA_DEFAULT_MODEL,
        timeout: float = OLLAMA_DEFAULT_TIMEOUT,
    ) -> None:
        """Initialize Ollama adapter.

        Args:
            host: Ollama server URL. Default: http://localhost:11434
            model: Model name to use. Default: llama3.1:8b
            timeout: Request timeout [s]. Default: 120.0
        """
        self._host = host.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._client: Any = None  # Lazy-loaded httpx client

        logger.info(
            "Initialized OllamaAdapter: host=%s, model=%s",
            self._host,
            self._model,
        )

    def _get_client(self) -> Any:
        """Get or create HTTP client.

        Lazy-loads httpx to avoid import errors if not installed.

        Returns:
            httpx.Client instance.

        Raises:
            AIProviderError: If httpx is not installed.
        """
        if self._client is None:
            try:
                import httpx

                self._client = httpx.Client(timeout=self._timeout)
            except ImportError as e:
                raise AIProviderError(
                    "httpx package required for OllamaAdapter. "
                    "Install with: pip install httpx",
                    provider="ollama",
                ) from e
        return self._client

    def send_message(
        self,
        message: str,
        context: ConversationContext,
        tools: list[ToolDeclaration],
    ) -> AgentResponse:
        """Send a message to local Ollama instance.

        Args:
            message: User message to process.
            context: Current conversation context.
            tools: Available tools for this request.

        Returns:
            AgentResponse with model's reply.

        Raises:
            AIConnectionError: If Ollama server is unreachable.
            AITimeoutError: If request times out.
            AIProviderError: For other Ollama errors.
        """
        client = self._get_client()

        # Format messages for Ollama
        messages = self._format_messages(context, message, tools)

        try:
            response = client.post(
                f"{self._host}/api/chat",
                json={
                    "model": self._model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                    },
                },
            )
            response.raise_for_status()

        except Exception as e:
            import httpx

            if isinstance(e, httpx.ConnectError):
                raise AIConnectionError(
                    f"Cannot connect to Ollama at {self._host}. "
                    "Is Ollama running? Start with: ollama serve",
                    provider="ollama",
                ) from e
            if isinstance(e, httpx.TimeoutException):
                raise AITimeoutError(
                    f"Ollama request timed out after {self._timeout}s",
                    provider="ollama",
                    timeout=self._timeout,
                ) from e
            raise AIProviderError(
                f"Ollama error: {e}",
                provider="ollama",
            ) from e

        # Parse response
        data = response.json()
        return self._parse_response(data)

    def stream_response(
        self,
        message: str,
        context: ConversationContext,
        tools: list[ToolDeclaration],
    ) -> Iterator[AgentChunk]:
        """Stream response chunks from Ollama.

        Args:
            message: User message to process.
            context: Current conversation context.
            tools: Available tools.

        Yields:
            AgentChunk instances as they arrive.
        """
        client = self._get_client()
        messages = self._format_messages(context, message, tools)

        try:
            with client.stream(
                "POST",
                f"{self._host}/api/chat",
                json={
                    "model": self._model,
                    "messages": messages,
                    "stream": True,
                },
            ) as response:
                response.raise_for_status()

                index = 0
                for line in response.iter_lines():
                    if not line:
                        continue

                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    is_done = data.get("done", False)

                    yield AgentChunk(
                        content=content,
                        is_final=is_done,
                        index=index,
                    )
                    index += 1

        except Exception as e:
            logger.error("Ollama streaming error: %s", e)
            raise AIProviderError(
                f"Ollama streaming error: {e}",
                provider="ollama",
            ) from e

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Return Ollama capabilities.

        Returns:
            ProviderCapabilities for Ollama.
        """
        # Note: Function calling support varies by model
        # llama3.1 and newer support it
        supported = frozenset(
            {
                ProviderCapability.STREAMING,
                ProviderCapability.SYSTEM_MESSAGE,
            }
        )

        # Check if model likely supports function calling
        if any(x in self._model.lower() for x in ["llama3", "mistral"]):
            supported = supported | frozenset({ProviderCapability.FUNCTION_CALLING})

        return ProviderCapabilities(
            supported=supported,
            max_tokens=8192,  # Varies by model
            model_name=self._model,
            provider_name="ollama",
        )

    def validate_connection(self) -> tuple[bool, str]:
        """Test connection to local Ollama.

        Verifies:
        1. Ollama server is running
        2. Configured model is available

        Returns:
            Tuple of (success, diagnostic_message).
        """
        try:
            client = self._get_client()

            # Check if Ollama is running
            response = client.get(f"{self._host}/api/tags")

            if response.status_code != 200:
                return False, f"Ollama returned status {response.status_code}"

            # Check if model is available
            data = response.json()
            models = data.get("models", [])
            model_names = [m.get("name", "") for m in models]

            # Handle model name with/without tag
            model_base = self._model.split(":")[0]
            available = any(m.startswith(model_base) for m in model_names)

            if not available:
                if not model_names:
                    return False, (
                        f"No models installed. Pull one with: ollama pull {self._model}"
                    )
                return False, (
                    f"Model '{self._model}' not found. "
                    f"Available: {', '.join(model_names[:5])}"
                )

            return True, f"Connected to Ollama with {self._model}"

        except AIProviderError:
            return False, ("httpx not installed. Install with: pip install httpx")
        except Exception as e:
            import httpx

            if isinstance(e, httpx.ConnectError):
                return False, (
                    f"Cannot connect to Ollama at {self._host}. "
                    "Is it running? Start with: ollama serve"
                )
            return False, f"Connection error: {e}"

    def _format_messages(
        self,
        context: ConversationContext,
        current_message: str,
        tools: list[ToolDeclaration],
    ) -> list[dict[str, str]]:
        """Format messages for Ollama API.

        Args:
            context: Conversation context.
            current_message: Current user message.
            tools: Available tools.

        Returns:
            List of message dicts for Ollama.
        """
        messages: list[dict[str, str]] = []

        # Add system prompt
        system_prompt = self.build_system_prompt(
            tools,
            context.user_expertise.name.lower(),
        )
        messages.append(
            {
                "role": "system",
                "content": system_prompt,
            }
        )

        # Add conversation history
        for msg in context.messages:
            messages.append(
                {
                    "role": msg.role if msg.role != "tool" else "assistant",
                    "content": msg.content,
                }
            )

        # Add current message
        messages.append(
            {
                "role": "user",
                "content": current_message,
            }
        )

        return messages

    def _parse_response(self, data: dict[str, Any]) -> AgentResponse:
        """Parse Ollama response into AgentResponse.

        Args:
            data: Raw response from Ollama.

        Returns:
            Parsed AgentResponse.
        """
        message = data.get("message", {})
        content = message.get("content", "")

        # Parse tool calls if present (model-dependent)
        tool_calls: list[ToolCall] = []
        if "tool_calls" in message:
            for tc in message["tool_calls"]:
                tool_calls.append(
                    ToolCall(
                        id=tc.get("id", f"tc_{len(tool_calls)}"),
                        name=tc.get("function", {}).get("name", ""),
                        arguments=tc.get("function", {}).get("arguments", {}),
                    )
                )

        # Extract usage if available
        usage: dict[str, int] = {}
        if "prompt_eval_count" in data:
            usage["prompt_tokens"] = data["prompt_eval_count"]
        if "eval_count" in data:
            usage["completion_tokens"] = data["eval_count"]

        return AgentResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason="stop" if data.get("done", True) else "length",
            usage=usage,
            metadata={
                "model": data.get("model", self._model),
                "total_duration": data.get("total_duration"),
            },
        )

    def list_available_models(self) -> list[str]:
        """List models available in the local Ollama instance.

        Returns:
            List of model names.

        Raises:
            AIConnectionError: If Ollama is not reachable.
        """
        try:
            client = self._get_client()
            response = client.get(f"{self._host}/api/tags")
            response.raise_for_status()

            data = response.json()
            return [m.get("name", "") for m in data.get("models", [])]

        except Exception as e:
            raise AIConnectionError(
                f"Cannot list Ollama models: {e}",
                provider="ollama",
            ) from e

    def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama library.

        This is a blocking operation that can take several minutes
        for large models.

        Args:
            model_name: Name of model to pull (e.g., 'llama3.1:8b').

        Returns:
            True if pull succeeded.

        Raises:
            AIProviderError: If pull fails.
        """
        try:
            # Verify httpx is available (will be used by download_client)
            self._get_client()

            # Long timeout for model downloads
            import httpx

            with httpx.Client(timeout=3600.0) as download_client:
                response = download_client.post(
                    f"{self._host}/api/pull",
                    json={"name": model_name},
                )
                response.raise_for_status()
                return True

        except Exception as e:
            raise AIProviderError(
                f"Failed to pull model {model_name}: {e}",
                provider="ollama",
            ) from e
