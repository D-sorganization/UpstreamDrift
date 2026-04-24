"""Headless assistant core for provider-agnostic AI sessions.

This module provides :class:`AssistantSession`, a pure-Python (no PyQt)
session class that can be embedded in web dashboards, CLI tools, or any
other integration that does not need a desktop GUI.

Streaming protocol
------------------
``send_message`` is an async generator that yields ``str`` chunks exactly
as they arrive from the underlying adapter.  Consumers can reassemble the
full response by concatenating all yielded chunks:

.. code-block:: python

    full = ""
    async for chunk in session.send_message("Hello"):
        full += chunk

Each chunk is a raw text fragment (never ``None``).  The generator
completes (``StopAsyncIteration``) when the provider signals the end of
the stream.

Tool confirmation gate
----------------------
Tools registered with ``requires_confirmation=True`` are **not** executed
automatically.  Instead, ``send_message`` calls the optional
``confirmation_callback`` supplied at construction time:

.. code-block:: python

    async def my_confirm(tool_name: str, arguments: dict) -> bool:
        answer = input(f"Allow {tool_name}? [y/N] ")
        return answer.lower() == "y"

    session = AssistantSession(
        provider="anthropic",
        confirmation_callback=my_confirm,
    )

If no callback is provided the tool is silently skipped and a notice is
appended to the stream.

WebSocket compatibility
-----------------------
The streaming protocol is intentionally identical to the one used by
``src/api/routes/chat_ws.py``: plain text chunks terminated by the
generator returning.  Dashboard integrations can pipe ``send_message``
output directly into a WebSocket send loop.
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
from collections.abc import AsyncIterator, Callable, Coroutine
from typing import Any

from src.shared.python.ai.tool_registry import ToolRegistry
from src.shared.python.ai.types import ConversationContext, Message
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

_ConfirmCallback = Callable[[str, dict[str, Any]], Coroutine[Any, Any, bool]]


def _build_adapter(provider: str, **kwargs: Any) -> Any:
    """Instantiate a provider adapter by name.

    Args:
        provider: One of ``"ollama"``, ``"openai"``, ``"anthropic"``,
            ``"gemini"``.
        **kwargs: Forwarded to the adapter constructor (e.g. ``api_key``,
            ``model``, ``host``).

    Returns:
        A :class:`~src.shared.python.ai.adapters.base.BaseAgentAdapter`
        instance.

    Raises:
        ValueError: If *provider* is not recognised.
        ImportError: If the adapter module is not available.
    """
    provider_lower = provider.lower()
    if provider_lower == "ollama":
        from src.shared.python.ai.adapters.ollama_adapter import OllamaAdapter

        return OllamaAdapter(**kwargs)
    if provider_lower == "openai":
        from src.shared.python.ai.adapters.openai_adapter import OpenAIAdapter

        return OpenAIAdapter(**kwargs)
    if provider_lower == "anthropic":
        from src.shared.python.ai.adapters.anthropic_adapter import AnthropicAdapter

        return AnthropicAdapter(**kwargs)
    if provider_lower == "gemini":
        from src.shared.python.ai.adapters.gemini_adapter import GeminiAdapter

        return GeminiAdapter(**kwargs)
    raise ValueError(
        f"Unknown provider '{provider}'. "
        "Supported values: 'ollama', 'openai', 'anthropic', 'gemini'."
    )


class AssistantSession:
    """Provider-agnostic, headless AI assistant session.

    This class manages a single conversation context and streams responses
    from a configured AI provider.  It has **no** dependency on PyQt or any
    other GUI toolkit and is safe to use in web servers, CLI tools, and
    automated tests.

    Parameters
    ----------
    provider:
        Name of the AI provider adapter to load.  Supported values are
        ``"ollama"`` (default), ``"openai"``, ``"anthropic"``, ``"gemini"``.
    adapter:
        Pass a pre-configured adapter instance directly.  When given,
        *provider* and *adapter_kwargs* are ignored.
    adapter_kwargs:
        Keyword arguments forwarded to the adapter constructor (e.g.
        ``api_key="sk-…"``, ``model="gpt-4o"``).
    tool_registry:
        A :class:`~src.shared.python.ai.tool_registry.ToolRegistry` whose
        registered tools the assistant may invoke.  Pass ``None`` (the
        default) to disable tool use entirely.
    rag_store:
        Optional RAG store used to retrieve context snippets before each
        response.  Must implement a ``query(text, top_k=5)`` method that
        returns ``list[tuple[Document, float]]``.  Pass ``None`` to skip
        RAG retrieval.
    confirmation_callback:
        Async callable invoked before executing tools that have
        ``requires_confirmation=True``.  Signature:
        ``async (tool_name: str, arguments: dict) -> bool``.
        If ``None``, confirmation-gated tools are skipped automatically.
    system_prompt:
        Override the default system prompt inserted at the start of every
        conversation.

    Example
    -------
    .. code-block:: python

        session = AssistantSession(provider="anthropic", adapter_kwargs={
            "api_key": "sk-ant-…",
            "model": "claude-3-5-sonnet-20241022",
        })
        async for chunk in session.send_message("What engines are available?"):
            print(chunk, end="", flush=True)
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are an AI assistant for the Golf Modeling Suite, a research-grade "
        "biomechanics simulation platform. Help users analyse golf swings using "
        "physics simulations across MuJoCo, Drake, and Pinocchio engines. "
        "Be concise, accurate, and cite uncertainty where relevant."
    )

    def __init__(
        self,
        provider: str = "ollama",
        *,
        adapter: Any = None,
        adapter_kwargs: dict[str, Any] | None = None,
        tool_registry: ToolRegistry | None = None,
        rag_store: Any | None = None,
        confirmation_callback: _ConfirmCallback | None = None,
        system_prompt: str | None = None,
    ) -> None:
        if adapter is not None:
            self._adapter = adapter
        else:
            self._adapter = _build_adapter(provider, **(adapter_kwargs or {}))

        self._tool_registry: ToolRegistry | None = tool_registry
        self._rag_store = rag_store
        self._confirmation_callback = confirmation_callback
        self._system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self._context = ConversationContext()
        logger.info(
            "AssistantSession ready (provider=%s, tools=%s, rag=%s)",
            provider,
            "yes" if tool_registry else "no",
            "yes" if rag_store else "no",
        )

    def get_history(self) -> list[Message]:
        """Return the current conversation history.

        Returns
        -------
        list[Message]
            Ordered list of all messages exchanged in this session.
        """
        return list(self._context.messages)

    def reset(self) -> None:
        """Clear conversation history, starting a fresh session.

        The adapter, tool registry, and RAG store are retained.
        """
        self._context = ConversationContext()
        logger.debug("AssistantSession: conversation reset")

    async def send_message(self, text: str) -> AsyncIterator[str]:
        """Send a user message and stream the assistant's response.

        This is an async generator.  Each ``yield`` delivers a plain-text
        chunk from the provider as soon as it is available, enabling
        real-time display.

        Streaming protocol
        ~~~~~~~~~~~~~~~~~~
        * Chunks are non-empty strings.
        * Tool execution notices are interspersed as
          ``"\\n[Tool <name>: <result>]\\n"`` fragments.
        * The generator returns (``StopAsyncIteration``) when the provider
          signals completion.

        Parameters
        ----------
        text:
            The user's message.  Must be a non-empty string.

        Yields
        ------
        str
            Text chunks from the assistant response stream.

        Raises
        ------
        ValueError
            If *text* is empty.
        """
        if not text or not text.strip():
            raise ValueError("text must be a non-empty string")

        self._context.add_user_message(text)

        rag_snippets = await self._fetch_rag_snippets(text)
        temp_ctx = self._build_temp_context(rag_snippets)

        tool_declarations = (
            self._tool_registry.declarations()
            if self._tool_registry is not None
            else []
        )

        chunk_q: queue.Queue[str | None] = queue.Queue()
        full_response: list[str] = []

        def _stream_sync() -> None:
            """Run the synchronous adapter stream in a worker thread."""
            try:
                pending: dict[int, dict[str, Any]] = {}

                for chunk in self._adapter.stream_response(
                    "",
                    temp_ctx,
                    tool_declarations,
                ):
                    if chunk.content:
                        chunk_q.put(chunk.content)
                        full_response.append(chunk.content)

                    if chunk.tool_call_delta:
                        for tc in chunk.tool_call_delta.get("tool_calls", []):
                            idx = tc.get("index", 0)
                            if idx not in pending:
                                pending[idx] = {"id": "", "name": "", "arguments": ""}
                            entry = pending[idx]
                            if tc.get("id"):
                                entry["id"] = tc["id"]
                            fn = tc.get("function") or {}
                            if fn.get("name"):
                                entry["name"] = fn["name"]
                            if fn.get("arguments"):
                                entry["arguments"] += fn["arguments"]

                    if chunk.is_final and pending:
                        self._flush_tool_calls(pending, chunk_q, full_response)
                        pending.clear()

            except (RuntimeError, ValueError, OSError) as exc:
                chunk_q.put(f"\n[Error: {exc}]")
            finally:
                chunk_q.put(None)

        worker = threading.Thread(target=_stream_sync, daemon=True)
        worker.start()

        while True:
            try:
                item = await asyncio.to_thread(chunk_q.get, timeout=60.0)
            except (queue.Empty, OSError):
                break
            if item is None:
                break
            yield item

        worker.join(timeout=5.0)

        complete = "".join(full_response)
        if complete:
            self._context.add_assistant_message(complete)
            logger.debug("AssistantSession: response saved (%d chars)", len(complete))

    def _flush_tool_calls(
        self,
        pending: dict[int, dict[str, Any]],
        chunk_q: queue.Queue[str | None],
        full_response: list[str],
    ) -> None:
        """Execute accumulated tool calls and push result notices into the queue.

        Args:
            pending: Accumulated tool-call deltas keyed by stream index.
            chunk_q: Output queue receiving text chunks.
            full_response: Accumulator for the complete response string.
        """
        if self._tool_registry is None:
            return

        for entry in pending.values():
            tool_name = entry["name"]
            tool_call_id = entry["id"] or f"tc_{tool_name}"
            try:
                arguments = json.loads(entry["arguments"]) if entry["arguments"] else {}
            except json.JSONDecodeError:
                arguments = {}

            tool_obj = self._tool_registry.get_tool(tool_name)
            if tool_obj is None:
                logger.warning("Tool not found in registry: %s", tool_name)
                continue

            if tool_obj.requires_confirmation and self._confirmation_callback is None:
                notice = f"\n[Tool {tool_name}: skipped (confirmation required)]\n"
                chunk_q.put(notice)
                full_response.append(notice)
                logger.info(
                    "Tool '%s' skipped — no confirmation_callback configured",
                    tool_name,
                )
                continue

            if (
                tool_obj.requires_confirmation
                and self._confirmation_callback is not None
            ):
                approved = self._run_confirmation(tool_name, arguments)
                if not approved:
                    notice = f"\n[Tool {tool_name}: rejected by user]\n"
                    chunk_q.put(notice)
                    full_response.append(notice)
                    continue

            logger.info("Executing tool '%s' (id=%s)", tool_name, tool_call_id)
            try:
                result = self._tool_registry.execute(tool_name, arguments, tool_call_id)
                result_content = json.dumps(result.to_dict())
            except Exception as exc:  # noqa: BLE001
                logger.warning("Tool execution failed: %s: %s", tool_name, exc)
                result_content = json.dumps({"success": False, "error": str(exc)})

            self._context.add_tool_result(tool_call_id, result_content)
            notice = f"\n[Tool {tool_name}: {result_content}]\n"
            chunk_q.put(notice)
            full_response.append(notice)

    def _run_confirmation(self, tool_name: str, arguments: dict[str, Any]) -> bool:
        """Run the async confirmation callback from a sync context.

        Args:
            tool_name: Name of the tool requesting approval.
            arguments: Arguments the tool will receive.

        Returns:
            True if the user approved, False otherwise.
        """
        if self._confirmation_callback is None:
            return False
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                self._confirmation_callback(tool_name, arguments)
            )
        except RuntimeError:
            return asyncio.run(self._confirmation_callback(tool_name, arguments))

    async def _fetch_rag_snippets(self, text: str) -> str:
        """Retrieve relevant RAG snippets for a query.

        Args:
            text: User query text.

        Returns:
            Formatted string of retrieved snippets, or empty string if none.
        """
        if self._rag_store is None:
            return ""
        try:
            results = await asyncio.to_thread(self._rag_store.query, text, 5)
            if not results:
                return ""
            snippets = "\n\n".join(
                f"[{doc.metadata.get('source', doc.id)}]\n{doc.content[:400]}"
                for doc, _score in results
            )
            return "Relevant documentation:\n\n" + snippets
        except Exception as exc:  # noqa: BLE001
            logger.warning("RAG query failed: %s", exc)
            return ""

    def _build_temp_context(self, rag_snippets: str) -> ConversationContext:
        """Build a temporary context copy with system prompt and RAG context injected.

        Args:
            rag_snippets: Pre-fetched RAG context string (may be empty).

        Returns:
            A :class:`ConversationContext` with system messages prepended.
        """
        from src.shared.python.ai.types import Message

        parts: list[str] = [self._system_prompt]
        if rag_snippets:
            parts.append(rag_snippets)

        system_msg = Message(role="system", content="\n\n".join(parts))
        temp_messages = [system_msg, *self._context.messages]

        return ConversationContext(
            session_id=self._context.session_id,
            messages=temp_messages,
            user_expertise=self._context.user_expertise,
            metadata=dict(self._context.metadata),
        )
