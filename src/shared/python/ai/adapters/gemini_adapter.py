# ruff: noqa: E501
"""Google Gemini API Adapter.

This module provides the adapter interface for Google's Gemini models
via the google-generativeai library.

Thread-safety / multi-instance notes (issue #2756)
--------------------------------------------------
The legacy ``google-generativeai`` SDK (versions <0.5) exposes
``genai.configure(api_key=...)`` as the *only* documented way to provide
credentials. This sets a process-global API key inside the ``genai`` module:
constructing two ``GeminiAdapter`` instances with different keys would silently
clobber each other.

Newer SDK versions (0.5+) ship a per-instance ``genai.Client`` object that
accepts ``api_key=...`` directly. We prefer that path when available and fall
back to a ``threading.RLock`` that re-applies ``genai.configure(...)`` right
before every request, so concurrent adapters with different keys cannot race.

Tool / function-calling notes (issue #2764)
-------------------------------------------
``send_message`` and ``stream_response`` historically accepted a ``tools``
parameter but silently dropped it: nothing was ever forwarded to Gemini's
function-calling API. To make the contract honest we now:

  * advertise ``FUNCTION_CALLING`` as **unsupported** in :pyattr:`capabilities`,
  * raise :class:`NotImplementedError` if a caller passes a non-empty
    ``tools`` list, and
  * emit a loud ``logger.warning`` so misconfigured callers (e.g. the
    assistant panel) surface the problem in logs.

A future PR (TODO(#2764)) should implement option A: translate
:class:`ToolDeclaration` into Gemini ``FunctionDeclaration`` objects and
handle function-call response parts. See
https://github.com/D-sorganization/Tools/issues/2764 for design notes.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.shared.python.ai.types import ChatModelInfo

from src.shared.python.ai.adapters.base import BaseAgentAdapter, ToolDeclaration
from src.shared.python.ai.types import (
    AgentChunk,
    AgentResponse,
    ConversationContext,
    ProviderCapabilities,
)
from src.shared.python.contracts import precondition
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

# Try to import google-generativeai. The newer ``Client`` symbol is only
# available on SDK 0.5+. We import it conditionally so the legacy fallback
# still works on the pinned ``>=0.3.0``.
try:
    import google.generativeai as genai
    from google.generativeai import GenerativeModel
    from google.generativeai.types import GenerateContentResponse

    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    from google.generativeai import Client as _GenaiClient  # type: ignore[attr-defined]

    HAS_GEMINI_CLIENT = True
except (ImportError, AttributeError):
    _GenaiClient = None  # type: ignore[assignment,misc]
    HAS_GEMINI_CLIENT = False


# Global re-configure lock used by the legacy fallback. A module-level
# ``RLock`` is the right scope because ``genai.configure`` itself mutates
# module-level state in the third-party SDK.
_CONFIGURE_LOCK: threading.RLock = threading.RLock()


class GeminiAdapter(BaseAgentAdapter):
    """Adapter for Google Gemini API.

    Each instance keeps its own ``api_key`` and serializes access to the
    third-party SDK's process-global state, so two adapters with different
    keys never clobber each other (issue #2756).

    The adapter does not currently implement function-calling. Passing a
    non-empty ``tools`` argument raises :class:`NotImplementedError` rather
    than silently dropping the tools (issue #2764).
    """

    def __init__(self, api_key: str, model: str = "gemini-pro") -> None:
        """Initialize Gemini adapter.

        Args:
            api_key: Google Cloud / AI Studio API Key.
            model: Model identifier (e.g., 'gemini-pro').

        Raises:
            ImportError: If ``google-generativeai`` is not installed.
            ValueError: If ``api_key`` is empty.
        """
        if not HAS_GEMINI:
            raise ImportError(
                "google-generativeai package is not installed. "
                "Run `pip install google-generativeai`."
            )
        if not api_key or not api_key.strip():
            raise ValueError("api_key must be a non-empty string")

        self._api_key = api_key
        self._model_name = model

        # Prefer the newer per-instance Client API (SDK 0.5+) which avoids
        # the global-configure footgun described in issue #2756.
        if HAS_GEMINI_CLIENT and _GenaiClient is not None:
            self._client: Any | None = _GenaiClient(api_key=self._api_key)
            # On modern SDKs the Client owns the model factory. We keep a
            # bound model handle for parity with the legacy path.
            self._model = self._client.models.get(self._model_name)  # type: ignore[attr-defined]
        else:
            # Legacy fallback: re-apply configure() under a lock before each
            # request (see ``_with_configured_sdk``). We still construct the
            # model eagerly so ``validate_connection`` has something to call.
            self._client = None
            with _CONFIGURE_LOCK:
                genai.configure(api_key=self._api_key)
                self._model = GenerativeModel(self._model_name)

    # ------------------------------------------------------------------ #
    # Legacy-SDK helper                                                  #
    # ------------------------------------------------------------------ #
    def _with_configured_sdk(self) -> None:
        """Re-apply this instance's API key to the global SDK config.

        Only used on legacy SDK versions (<0.5) that do not expose a
        per-instance ``Client``. Callers must hold ``_CONFIGURE_LOCK`` for
        the duration of any request that follows this call.
        """
        if self._client is not None:
            return
        genai.configure(api_key=self._api_key)

    # ------------------------------------------------------------------ #
    # Tool-arg validation (issue #2764)                                  #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _reject_tools_if_present(tools: Sequence[ToolDeclaration] | None) -> None:
        """Refuse non-empty tool lists with a loud warning.

        TODO(#2764): replace this with a real translation from
        :class:`ToolDeclaration` to Gemini ``FunctionDeclaration`` and wire
        function-call response parts back through :class:`AgentResponse`.
        See https://github.com/D-sorganization/Tools/issues/2764.
        """
        if not tools:
            return
        tool_names = ", ".join(getattr(t, "name", "<unknown>") for t in tools)
        logger.warning(
            "GeminiAdapter received tools=[%s] but function-calling is not "
            "implemented (see issue #2764). Refusing the request rather than "
            "silently dropping the tools.",
            tool_names,
        )
        raise NotImplementedError(
            "GeminiAdapter does not support function-calling yet (issue #2764). "
            "Pass tools=[] or use a provider whose capabilities include "
            "FUNCTION_CALLING."
        )

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    @precondition(
        lambda message: bool(message.strip()), "message must not be empty or blank"
    )
    def send_message(
        self,
        message: str,
        context: ConversationContext,
        tools: list[ToolDeclaration],
    ) -> AgentResponse:
        """Send a message to Gemini.

        Raises:
            NotImplementedError: If ``tools`` is non-empty (see issue #2764).
        """
        self._reject_tools_if_present(tools)
        # Canonical zero-usage: Gemini SDK (v0.3+) does not expose per-call
        # token counts in a stable way, so we emit zeros rather than omitting
        # the key (issue #2763).
        canonical_usage = self._normalize_token_counts({})
        try:
            with _CONFIGURE_LOCK:
                self._with_configured_sdk()
                chat = self._build_chat_session(context)
                response = chat.send_message(message)
            return AgentResponse(content=response.text, usage=canonical_usage)
        except (RuntimeError, ValueError, OSError) as e:
            logger.error(f"Gemini API error: {e}")
            return AgentResponse(content=f"Error: {e}", usage=canonical_usage)

    def stream_response(
        self,
        message: str,
        context: ConversationContext,
        tools: list[ToolDeclaration],
    ) -> Iterator[AgentChunk]:
        """Stream response from Gemini.

        Raises:
            NotImplementedError: If ``tools`` is non-empty (see issue #2764).
        """
        self._reject_tools_if_present(tools)
        # Track whether we have emitted a final chunk so the guarantee below
        # can synthesize one if the underlying generator finishes without one
        # (issue #2763 — Gemini streaming finality fix).
        emitted_final = False
        try:
            with _CONFIGURE_LOCK:
                self._with_configured_sdk()
                chat = self._build_chat_session(context)
                response: Iterator[GenerateContentResponse] = chat.send_message(
                    message, stream=True
                )

                index = 0
                for chunk in response:
                    if chunk.text:
                        yield AgentChunk(
                            content=chunk.text, is_final=False, index=index
                        )
                        index += 1

        except (RuntimeError, ValueError, OSError) as e:
            logger.error(f"Gemini streaming error: {e}")
            yield AgentChunk(content=f"\n[Error: {e}]", is_final=True)
            emitted_final = True

        # Guarantee: every stream MUST end with is_final=True (issue #2763).
        if not emitted_final:
            yield AgentChunk(content="", is_final=True)

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Return the set of capabilities supported by the Gemini provider.

        ``FUNCTION_CALLING`` is intentionally **not** advertised: the adapter
        does not currently translate :class:`ToolDeclaration` into Gemini's
        function-calling format. Tracked in issue #2764.
        """
        from src.shared.python.ai.types import ProviderCapability

        return ProviderCapabilities(
            supported=frozenset(
                {
                    ProviderCapability.STREAMING,
                    ProviderCapability.VISION,
                }
            ),
            max_tokens=30720,  # Gemini 1.0 Pro context
            model_name=self._model_name,
            provider_name="google",
        )

    # ------------------------------------------------------------------ #
    # Tools issue #2871: provider catalogue + reasoning capabilities
    # ------------------------------------------------------------------ #

    _STATIC_MODELS: tuple[str, ...] = (
        "gemini-pro",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-1.0-pro",
    )

    def list_models(self) -> list[ChatModelInfo]:
        """Return Gemini model info; falls back to a static catalogue.

        The ``google-generativeai`` SDK exposes ``genai.list_models()``
        which performs a network call.  Since Gemini's process-global
        state makes that call expensive and unstable in unit tests, we
        prefer the static catalogue and never raise.
        """
        from src.shared.python.ai.types import ChatModelInfo

        # Network probe intentionally skipped — see docstring.
        return [
            ChatModelInfo(
                name=model_id,
                provider="google",
                display_name=model_id,
                id=model_id,
            )
            for model_id in self._STATIC_MODELS
        ]

    def thinking_capabilities(self) -> Any:
        """Gemini does not currently expose user-controllable thinking budgets."""
        from src.shared.python.chat.models import make_none_only_capabilities

        return make_none_only_capabilities(provider="google")

    def validate_connection(self) -> tuple[bool, str]:
        """Validate Gemini connection."""
        try:
            if not HAS_GEMINI:
                return False, "google-generativeai package missing"

            with _CONFIGURE_LOCK:
                self._with_configured_sdk()
                self._model.generate_content("Hello")
            return True, "Connected successfully"
        except (RuntimeError, ValueError, OSError) as e:
            logger.error(f"Gemini validation error: {e}")
            return False, f"Connection failed: {e}"

    def _build_chat_session(self, context: ConversationContext) -> Any:
        """Build a chat session with history."""
        if context is None:
            raise ValueError("context must be provided")
        history = []
        for msg in context.messages:
            role = "user" if msg.role == "user" else "model"
            history.append({"role": role, "parts": [msg.content]})

        return self._model.start_chat(history=history)  # type: ignore[arg-type]
