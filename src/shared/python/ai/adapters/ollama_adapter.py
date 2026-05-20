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
from typing import TYPE_CHECKING, Any

from src.shared.python.ai.adapters.base import BaseAgentAdapter, ToolDeclaration
from src.shared.python.ai.config import (
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_TIMEOUT,
    get_ollama_host,
    get_ollama_model,
    get_ollama_timeout,
)
from src.shared.python.ai.exceptions import (
    AIConnectionError,
    AIProviderError,
)
from src.shared.python.ai.types import (
    AgentChunk,
    AgentResponse,
    ConversationContext,
    ProviderCapabilities,
    ProviderCapability,
    ToolCall,
)
from src.shared.python.contracts import precondition
from src.shared.python.logging_pkg.logging_config import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = get_logger(__name__)

# Backwards compatibility aliases
OLLAMA_DEFAULT_HOST = DEFAULT_OLLAMA_HOST
OLLAMA_DEFAULT_MODEL = DEFAULT_OLLAMA_MODEL
OLLAMA_DEFAULT_TIMEOUT = DEFAULT_OLLAMA_TIMEOUT


def _normalize_host(host: str) -> str:
    """Coerce a user-provided Ollama host into a usable URL.

    The settings dialog and legacy configs sometimes store the host without a
    scheme (e.g. ``"0.0.0.0:11434"``, ``"localhost:11434"``). httpx requires a
    scheme and will raise ``UnsupportedProtocol`` otherwise, which surfaces in
    the chat UI as a generic connection failure. Defensive prepending of
    ``http://`` (the only protocol Ollama serves on by default) makes the
    adapter tolerant of these inputs.

    ``0.0.0.0`` is rewritten to ``127.0.0.1`` because it is a wildcard
    *listen* address, not a routable client address, and httpx will fail to
    connect to it on Windows.
    """
    if not host:
        return DEFAULT_OLLAMA_HOST
    host = host.strip().rstrip("/")
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    return host.replace("://0.0.0.0", "://127.0.0.1")


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
        host: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        """Initialize Ollama adapter.

        Configuration is loaded from environment variables if not provided:
            - OLLAMA_HOST: Server URL (default: http://localhost:11434)
            - OLLAMA_MODEL: Model name (default: llama3.1:8b)
            - OLLAMA_TIMEOUT: Timeout in seconds (default: 120.0)

        Args:
            host: Ollama server URL. Uses OLLAMA_HOST env var or default.
            model: Model name to use. Uses OLLAMA_MODEL env var or default.
            timeout: Request timeout [s]. Uses OLLAMA_TIMEOUT env var or default.
        """
        self._host = _normalize_host(host or get_ollama_host())
        self._model = model or get_ollama_model()
        self._timeout = timeout if timeout is not None else get_ollama_timeout()
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

    @precondition(
        lambda message: bool(message.strip()), "message must not be empty or blank"
    )
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
        if message is None:
            raise ValueError("message must be provided")
        if message is None:
            raise ValueError("message must be provided")
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

        except Exception as e:  # noqa: BLE001
            return self._handle_error(e)

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
        if message is None:
            raise ValueError("message must be provided")
        if message is None:
            raise ValueError("message must be provided")
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
            self._handle_error(e)

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

    # ------------------------------------------------------------------ #
    # Tools issue #2871: provider catalogue + reasoning capabilities
    # ------------------------------------------------------------------ #

    _STATIC_MODELS: tuple[str, ...] = (
        "llama3.1:8b",
        "llama3.1:70b",
        "llama3:8b",
        "mistral:7b",
        "qwen2:7b",
        "phi3:medium",
    )

    def list_models(self) -> list[ChatModelInfo]:
        """Return Ollama model info; falls back to a static catalogue."""
        from src.shared.python.ai.types import ChatModelInfo

        names: list[str] = []
        try:
            client = self._get_client()
            response = client.get(f"{self._host}/api/tags", timeout=1.0)
            response.raise_for_status()
            payload = response.json()
            models = payload.get("models") if isinstance(payload, dict) else None
            if isinstance(models, list):
                names = [
                    str(entry.get("name"))
                    for entry in models
                    if isinstance(entry, dict) and entry.get("name")
                ]
                names = [name for name in names if name.strip()]
        except Exception:  # noqa: BLE001 - any failure → static catalogue
            logger.debug(
                "Ollama list_models live probe failed; using static catalogue",
                exc_info=True,
            )

        if not names:
            names = list(self._STATIC_MODELS)

        return [
            ChatModelInfo(
                name=name,
                provider="ollama",
                display_name=name,
                id=name,
            )
            for name in names
        ]

    def thinking_capabilities(self) -> Any:
        """Ollama models do not currently expose reasoning budgets."""
        from src.shared.python.chat.models import make_none_only_capabilities

        return make_none_only_capabilities(provider="ollama")

    def validate_connection(self) -> tuple[bool, str]:
        """Test connection to local Ollama.

        Verifies:
        1. Ollama server is running
        2. Configured model is available; if not, auto-fall back to the first
           locally-installed model so chat keeps working when saved settings
           reference a model that has been removed.

        Returns:
            Tuple of (success, diagnostic_message).
        """
        try:
            client = self._get_client()

            # Check if Ollama is running
            response = client.get(f"{self._host}/api/tags", timeout=1.0)

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
                # Saved-settings model no longer exists locally. Prefer a real
                # (non-:cloud) model so chat stays offline-capable; fall back to
                # whatever is listed first if none qualify.
                fallback = next(
                    (m for m in model_names if not m.endswith(":cloud")),
                    model_names[0],
                )
                logger.warning(
                    "Configured Ollama model '%s' not installed; "
                    "auto-falling back to '%s'. Update saved settings to dismiss.",
                    self._model,
                    fallback,
                )
                self._model = fallback
                return True, (
                    f"Configured model not installed; using '{fallback}' instead. "
                    f"Available: {', '.join(model_names[:5])}"
                )

            return True, f"Connected to Ollama with {self._model}"

        except AIProviderError:
            return False, ("httpx not installed. Install with: pip install httpx")
        except Exception as e:  # noqa: BLE001
            # httpx is lazily imported; ConnectError cannot be listed statically.
            import httpx

            if isinstance(e, httpx.ConnectError):
                return False, (
                    f"Cannot connect to Ollama at {self._host}. "
                    "Is it running? Start with: ollama serve"
                )
            logger.debug("Ollama connection check failed: %s: %s", type(e).__name__, e)
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
        if context is None:
            raise ValueError("context must be provided")
        if context is None:
            raise ValueError("context must be provided")
        messages: list[dict[str, str]] = []

        # Add system prompt
        system_prompt = self.build_system_prompt(
            tools,
            context.user_expertise.name.lower(),
            context,
        )
        messages.append(
            {
                "role": "system",
                "content": system_prompt,
            }
        )

        # Add conversation history
        messages.extend(
            [
                {
                    "role": msg.role if msg.role != "tool" else "assistant",
                    "content": msg.content,
                }
                for msg in context.messages
            ]
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
        if data is None:
            raise ValueError("data must be provided")
        if data is None:
            raise ValueError("data must be provided")
        message = data.get("message", {})
        content = message.get("content", "")

        # Parse tool calls if present (model-dependent)
        tool_calls: list[ToolCall] = []
        if "tool_calls" in message:
            tool_calls.extend(
                [
                    ToolCall(
                        id=tc.get("id", f"tc_{len(tool_calls)}"),
                        name=tc.get("function", {}).get("name", ""),
                        arguments=tc.get("function", {}).get("arguments", {}),
                    )
                    for tc in message["tool_calls"]
                ]
            )

        # Extract usage and normalize to canonical keys (issue #2763).
        # Ollama uses ``prompt_eval_count`` / ``eval_count`` internally;
        # _normalize_token_counts maps prompt_tokens → input_tokens etc.
        raw_usage: dict[str, int] = {}
        if "prompt_eval_count" in data:
            raw_usage["prompt_tokens"] = data["prompt_eval_count"]
        if "eval_count" in data:
            raw_usage["completion_tokens"] = data["eval_count"]
        usage = self._normalize_token_counts(raw_usage)

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
            response = client.get(f"{self._host}/api/tags", timeout=1.0)
            response.raise_for_status()

            data = response.json()
            return [m.get("name", "") for m in data.get("models", [])]

        except Exception as e:  # noqa: BLE001
            # httpx errors (ConnectError, TimeoutException, HTTPStatusError, etc.)
            # are all subclasses of httpx.HTTPError or OSError; broad catch is
            # intentional here to convert any transport failure to AIConnectionError.
            logger.debug("Failed to list Ollama models: %s: %s", type(e).__name__, e)
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

        except Exception as e:  # noqa: BLE001
            self._handle_error(e)

    def _handle_error(self, error: Exception) -> AgentResponse:
        """Handle Ollama-specific errors before falling back to generic classifier."""
        err_str = str(error).lower()
        if "connection" in err_str or "unreachable" in err_str:
            raise AIConnectionError(
                f"Cannot connect to Ollama at {self._host}. "
                "Is Ollama running? Start with: ollama serve",
                provider="ollama",
            ) from error
        raise self._classify_error(
            error, provider="ollama", timeout=self._timeout
        ) from error
