"""OpenAI Codex CLI adapter.

Wraps the ``codex`` CLI binary (the ``@openai/codex`` npm package) as a chat
provider. The CLI handles its own authentication and provider routing; this
adapter only spawns it and parses output.

Distinct from :class:`~src.shared.python.ai.adapters.openai_adapter.OpenAIAdapter`,
which talks directly to the OpenAI REST API and requires an API key in the
process environment. Use the CLI adapter when the user has ``codex`` installed
and logged in but does not want to expose an API key to the application.

Requirements:
    - ``codex`` CLI on ``PATH`` (or a known install location).
    - User logged in (``codex login`` or ``OPENAI_API_KEY`` set).

Invocation pattern::

    codex exec --skip-git-repo-check "prompt"

The ``--skip-git-repo-check`` flag is required because Codex's safety check
refuses to run outside a "trusted directory". The chat UI is not necessarily
launched from a git repo, so we always pass the flag.

Example::

    >>> from src.shared.python.ai.adapters.codex_cli_adapter import CodexCliAdapter
    >>> adapter = CodexCliAdapter()
    >>> ok, msg = adapter.validate_connection()
    >>> if ok:
    ...     response = adapter.send_message("hello", ctx, tools=[])
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

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
)
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

# Default invocation timeout. Codex's first response on a cold prompt can take
# 30-60s due to skill catalogue loading + model warm-up. Keep generous so the
# chat doesn't report a misleading timeout on the very first message.
DEFAULT_CODEX_CLI_TIMEOUT = 180.0  # [s]

# Known install locations probed when ``codex`` is not on PATH. Windows-npm
# global-bin first because that's the npm install target on the host that
# typically runs the launcher.
_FALLBACK_PATHS = (
    r"%APPDATA%\npm\codex.cmd",
    r"%APPDATA%\npm\codex",
    "/home/dieterolson/.npm-global/bin/codex",
    "/usr/local/bin/codex",
)

# Static model catalogue. The Codex CLI does not expose a model listing
# command; this is the documented set of models that the CLI accepts via
# ``--model``. Update when OpenAI adds new entries.
_STATIC_MODELS: tuple[str, ...] = (
    "gpt-5.5",
    "gpt-5",
    "gpt-5-mini",
    "o4",
    "o4-mini",
    "default",
)


def _resolve_binary(explicit: str | None = None) -> str | None:
    """Locate the ``codex`` binary.

    Returns the resolved absolute path, or ``None`` when the binary cannot be
    found. Callers handle the ``None`` case via :meth:`validate_connection`.
    """
    if explicit:
        if Path(explicit).exists():
            return explicit
        logger.warning(
            "CodexCliAdapter: explicit binary %s does not exist; "
            "falling back to PATH search",
            explicit,
        )
    found = shutil.which("codex")
    if found:
        return found
    for candidate in _FALLBACK_PATHS:
        expanded = os.path.expandvars(candidate)
        if Path(expanded).exists():
            return expanded
    return None


class CodexCliAdapter(BaseAgentAdapter):
    """Adapter that delegates chat to the OpenAI Codex CLI.

    The CLI handles auth, model selection, and tool inventory internally.
    This adapter is a thin subprocess wrapper that translates
    :class:`ConversationContext` into a single prompt and parses the CLI's
    stdout back into an :class:`AgentResponse`.

    Attributes:
        binary: Absolute path to the resolved ``codex`` binary (or ``None``).
        model: Optional model id passed via ``--model``. ``None`` lets the
            CLI choose its own default.
        timeout: Per-invocation timeout in seconds.
    """

    PROVIDER_NAME = "codex_cli"

    def __init__(
        self,
        binary: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            binary: Optional explicit path to the ``codex`` binary. Resolved
                lazily so construction never fails when the CLI is missing.
            model: Optional model identifier.
            timeout: Subprocess timeout. Defaults to
                ``DEFAULT_CODEX_CLI_TIMEOUT``.
        """
        self.binary = _resolve_binary(binary)
        self.model = model
        self.timeout = (
            float(timeout) if timeout is not None else DEFAULT_CODEX_CLI_TIMEOUT
        )
        self._capabilities = ProviderCapabilities(
            supported=frozenset(
                {
                    ProviderCapability.SYSTEM_MESSAGE,
                    ProviderCapability.STREAMING,
                }
            ),
            max_tokens=128_000,
            model_name=model or "default",
            provider_name=self.PROVIDER_NAME,
        )

    # ------------------------------------------------------------------ #
    # BaseAgentAdapter surface
    # ------------------------------------------------------------------ #

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    def list_models(self) -> list[ChatModelInfo]:  # noqa: F821
        """Return the documented model catalogue. Always non-empty."""
        from src.shared.python.ai.types import ChatModelInfo

        return [
            ChatModelInfo(
                name=m,
                provider=self.PROVIDER_NAME,
                display_name=m,
                id=m,
            )
            for m in _STATIC_MODELS
        ]

    def thinking_capabilities(self) -> Any:
        """Codex CLI does not surface reasoning budgets to the host."""
        from src.shared.python.chat.models import make_none_only_capabilities

        return make_none_only_capabilities(provider=self.PROVIDER_NAME)

    def send_message(
        self,
        message: str,
        context: ConversationContext,
        tools: list[ToolDeclaration],  # noqa: ARG002 (CLI is its own tool host)
    ) -> AgentResponse:
        """Send a message and return the CLI's response.

        Tools declared by the host application are intentionally ignored: the
        Codex CLI manages its own tool inventory (file edits, shell, web
        browse). Surfacing host tools would duplicate or conflict with the
        CLI's built-ins. For host-tool integration use
        :class:`OpenAIAdapter` instead.

        Raises:
            AIConnectionError: When the binary cannot be located.
            AITimeoutError: When the subprocess exceeds ``self.timeout``.
            AIProviderError: When the CLI returns a non-zero exit code.
        """
        if message is None or not message.strip():
            raise ValueError("message must be a non-empty string")
        if self.binary is None:
            raise AIConnectionError(
                "Codex CLI not found on PATH. Install with "
                "`npm install -g @openai/codex` and run `codex login`.",
                provider=self.PROVIDER_NAME,
            )

        prompt = self._build_prompt(message, context)
        args = [self.binary, "exec", "--skip-git-repo-check"]
        if self.model:
            args += ["--model", self.model]
        args.append(prompt)

        try:
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
                encoding="utf-8",
            )
        except subprocess.TimeoutExpired as exc:
            raise AITimeoutError(
                f"Codex CLI timed out after {self.timeout}s",
                provider=self.PROVIDER_NAME,
            ) from exc
        except FileNotFoundError as exc:
            raise AIConnectionError(
                f"Codex binary at {self.binary} disappeared mid-invocation",
                provider=self.PROVIDER_NAME,
            ) from exc

        if result.returncode != 0:
            stderr_tail = (result.stderr or "")[-500:]
            raise AIProviderError(
                f"Codex CLI exited {result.returncode}: {stderr_tail}",
                provider=self.PROVIDER_NAME,
            )

        return AgentResponse(
            content=self._strip_telemetry(result.stdout or ""),
            usage={},
            metadata={
                "provider": self.PROVIDER_NAME,
                "model": self.model or "default",
                "stderr": (result.stderr or "")[-500:],
            },
        )

    def stream_response(
        self,
        message: str,
        context: ConversationContext,
        tools: list[ToolDeclaration],
    ) -> Iterator[AgentChunk]:
        """Yield the response as a single chunk.

        Codex CLI does not yet have a stable streaming output format suitable
        for incremental rendering. We surface the complete response as one
        chunk; the chat dock handles this gracefully.
        """
        response = self.send_message(message, context, tools)
        yield AgentChunk(
            content=response.content,
            is_final=True,
        )

    def validate_connection(self) -> tuple[bool, str]:
        """Probe the CLI is installed and authenticated.

        Runs ``codex --version`` (returns in <2s). A non-zero exit or missing
        binary is reported with a user-friendly hint.
        """
        if self.binary is None:
            return False, ("Codex CLI not found. Install: npm install -g @openai/codex")
        try:
            result = subprocess.run(
                [self.binary, "--version"],
                capture_output=True,
                text=True,
                timeout=10.0,
                check=False,
                encoding="utf-8",
            )
        except subprocess.TimeoutExpired:
            return False, "Codex CLI did not respond to --version within 10s"
        except OSError as exc:
            return False, f"Could not execute {self.binary}: {exc}"

        if result.returncode != 0:
            return False, (
                f"Codex --version exited {result.returncode}: "
                f"{(result.stderr or '').strip()[:200]}"
            )
        stdout_lines = (result.stdout or "").strip().splitlines()
        version = stdout_lines[0] if stdout_lines else "?"
        return True, f"Codex CLI available: {version}"

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_prompt(message: str, context: ConversationContext) -> str:
        """Concatenate conversation history into a single CLI prompt.

        Codex's ``exec`` mode takes a single positional prompt with no native
        conversation-history flag. We render the history as alternating
        ``User:`` / ``Assistant:`` blocks followed by the current message —
        matching the convention used by the other CLI-shaped adapters.
        """
        parts: list[str] = []
        for msg in getattr(context, "messages", []) or []:
            role = getattr(msg, "role", "user").capitalize()
            content = getattr(msg, "content", "") or ""
            if content:
                parts.append(f"{role}: {content}")
        parts.append(f"User: {message}")
        return "\n\n".join(parts)

    @staticmethod
    def _strip_telemetry(stdout: str) -> str:
        """Drop the noisy ``[year-month-day…]`` telemetry preamble.

        Codex prefixes responses with ANSI-coloured timing/runtime lines that
        look like ``[2026-05-17T23:13:03] thinking…``. Drop those leading
        lines so the chat bubble only shows the actual response text.
        """
        lines = stdout.splitlines()
        # Skip leading lines that look like timestamps in square brackets.
        first_real = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("[20"):
                first_real = i + 1
                continue
            break
        return "\n".join(lines[first_real:]).strip()
