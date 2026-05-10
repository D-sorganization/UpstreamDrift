"""Cline adapter for local IDE-based AI agent communication.

Cline runs as a local server (typically VS Code extension) and provides
an OpenAI-compatible REST API for chat and tool-calling.

Requirements:
    - Cline extension running in VS Code/Cursor
    - Local server accessible (default: http://localhost:3000)
    - httpx package: pip install httpx

Cost Model:
    - Free (uses Cline's configured provider under the hood)

Example::

    >>> from src.shared.python.ai.adapters.cline_adapter import ClineAdapter
    >>> adapter = ClineAdapter(host="http://localhost:3000")
    >>> success, msg = adapter.validate_connection()
    >>> if success:
    ...     response = adapter.send_message("Analyze this", context, tools)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
from src.shared.python.logging_pkg.logging_config import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = get_logger(__name__)

DEFAULT_CLINE_HOST = "http://localhost:3000"
DEFAULT_CLINE_TIMEOUT = 120.0  # [s]


class ClineAdapter(BaseAgentAdapter):
    """Adapter for Cline local IDE agent.

    Communicates with Cline's local OpenAI-compatible endpoint.
    Cline delegates to whichever provider it's configured with
    (Claude, GPT-4, local model, etc).

    Attributes:
        host: Cline server URL.
        timeout: Request timeout [s].
    """

    def __init__(
        self,
        host: str | None = None,
        timeout: float | None = None,
    ) -> None:
        """Initialize Cline adapter.

        Args:
            host: Cline server URL. Defaults to http://localhost:3000.
            timeout: Request timeout [s]. Defaults to 120.
        """
        self._host = (host or DEFAULT_CLINE_HOST).rstrip("/")
        self._timeout = timeout if timeout is not None else DEFAULT_CLINE_TIMEOUT
        self._client: Any = None

        logger.info("Initialized ClineAdapter: host=%s", self._host)

    def _get_client(self) -> Any:
        """Get or create httpx client.

        Returns:
            httpx.Client instance.

        Raises:
            AIProviderError: If httpx not installed.
        """
        if self._client is None:
            try:
                import httpx

                self._client = httpx.Client(
                    base_url=self._host,
                    timeout=self._timeout,
                )
            except ImportError as e:
                raise AIProviderError(
                    "httpx package required for ClineAdapter. "
                    "Install with: pip install httpx",
                    provider="cline",
                ) from e
        return self._client

    def send_message(
        self,
        message: str,
        context: ConversationContext,
        tools: list[ToolDeclaration],
    ) -> AgentResponse:
        """Send a message to Cline.

        Args:
            message: User message.
            context: Conversation context.
            tools: Available tools.

        Returns:
            AgentResponse with Cline's reply.
        """
        client = self._get_client()
        messages = self._format_messages(context, message)

        payload: dict[str, Any] = {
            "model": "cline",
            "messages": messages,
        }

        # Add tools in OpenAI format (Cline speaks OpenAI protocol)
        if tools:
            payload["tools"] = [t.to_openai_format() for t in tools]

        try:
            resp = client.post("/v1/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return self._parse_response(data)
        except Exception as e:
            return self._handle_error(e)

    def stream_response(
        self,
        message: str,
        context: ConversationContext,
        tools: list[ToolDeclaration],
    ) -> Iterator[AgentChunk]:
        """Stream response from Cline.

        Args:
            message: User message.
            context: Conversation context.
            tools: Available tools.

        Yields:
            AgentChunk instances.
        """
        client = self._get_client()
        messages = self._format_messages(context, message)

        payload: dict[str, Any] = {
            "model": "cline",
            "messages": messages,
            "stream": True,
        }
        if tools:
            payload["tools"] = [t.to_openai_format() for t in tools]

        try:
            import json as json_mod

            with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
                resp.raise_for_status()
                index = 0
                for line in resp.iter_lines():
                    if not line or line.startswith(":"):
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            yield AgentChunk(content="", is_final=True, index=index)
                            break
                        try:
                            data = json_mod.loads(data_str)
                            choices = data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield AgentChunk(
                                        content=content,
                                        is_final=False,
                                        index=index,
                                    )
                                    index += 1
                        except (json_mod.JSONDecodeError, KeyError):
                            continue

        except (RuntimeError, TypeError, ValueError, OSError) as e:
            logger.error("Cline streaming error: %s", e)
            raise AIProviderError(
                f"Cline streaming error: {e}",
                provider="cline",
            ) from e

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Return Cline capabilities.

        Returns:
            ProviderCapabilities for Cline.
        """
        return ProviderCapabilities(
            supported=frozenset(
                {
                    ProviderCapability.FUNCTION_CALLING,
                    ProviderCapability.STREAMING,
                    ProviderCapability.SYSTEM_MESSAGE,
                }
            ),
            max_tokens=128000,  # Depends on underlying provider
            model_name="cline",
            provider_name="cline",
        )

    def validate_connection(self) -> tuple[bool, str]:
        """Test connection to Cline server.

        Returns:
            Tuple of (success, diagnostic_message).
        """
        try:
            client = self._get_client()
            resp = client.get("/v1/models")
            if resp.status_code == 200:
                return True, f"Connected to Cline at {self._host}"
            return False, f"Cline returned status {resp.status_code}"
        except AIProviderError:
            return False, "httpx not installed"
        except (ConnectionError, TimeoutError, OSError) as e:
            return False, f"Cannot connect to Cline at {self._host}: {e}"

    def _format_messages(
        self,
        context: ConversationContext,
        current_message: str,
    ) -> list[dict[str, Any]]:
        """Format messages for OpenAI-compatible API.

        Args:
            context: Conversation context.
            current_message: Current user message.

        Returns:
            List of message dicts.
        """
        messages: list[dict[str, Any]] = []

        for msg in context.messages:
            entry: dict[str, Any] = {"role": msg.role, "content": msg.content}
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": str(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            if msg.tool_call_id:
                entry["tool_call_id"] = msg.tool_call_id
            messages.append(entry)

        messages.append({"role": "user", "content": current_message})
        return messages

    def _parse_response(self, data: dict[str, Any]) -> AgentResponse:
        """Parse OpenAI-format response.

        Args:
            data: Raw response data.

        Returns:
            Parsed AgentResponse.
        """
        choices = data.get("choices", [])
        if not choices:
            return AgentResponse(content="No response from Cline")

        choice = choices[0]
        message = choice.get("message", {})
        content = message.get("content", "")

        tool_calls: list[ToolCall] = []
        for tc_data in message.get("tool_calls", []):
            func = tc_data.get("function", {})
            import json as json_mod

            try:
                args = json_mod.loads(func.get("arguments", "{}"))
            except (json_mod.JSONDecodeError, TypeError):
                args = {}
            tool_calls.append(
                ToolCall(
                    id=tc_data.get("id", ""),
                    name=func.get("name", ""),
                    arguments=args,
                )
            )

        usage = data.get("usage", {})
        return AgentResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason", "stop"),
            usage={
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            },
            metadata={"model": data.get("model", "cline")},
        )

    def _handle_error(self, error: Exception) -> AgentResponse:
        """Handle Cline errors.

        Args:
            error: The exception.

        Raises:
            Appropriate AIError subclass.
        """
        error_str = str(error).lower()

        if "timeout" in error_str:
            raise AITimeoutError(
                f"Cline request timed out after {self._timeout}s",
                provider="cline",
                timeout=self._timeout,
            ) from error

        if "connection" in error_str or "refused" in error_str:
            raise AIConnectionError(
                f"Cannot connect to Cline at {self._host}. "
                "Ensure VS Code with Cline extension is running.",
                provider="cline",
            ) from error

        raise AIProviderError(
            f"Cline error: {error}",
            provider="cline",
        ) from error
