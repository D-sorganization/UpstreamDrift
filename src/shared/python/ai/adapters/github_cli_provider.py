"""GitHub CLI agent provider — wraps ``gh`` as a chat-level CLI agent.

Tools issue: #2899.

Complementary to the GitHub MCP server (Tools #2897). Both ship side-by-side
because they serve different user intents:

- **MCP server** (Tools #2897): the model calls structured GitHub tools
  (``list_issues``, ``create_issue``, ...) with JSON payloads. Programmatic,
  deterministic, tool-call-shaped.
- **CLI provider** (this module): the chat talks to ``gh`` like a shell user.
  The chat history shows ``$ gh issue list --me`` and its raw output.
  Natural for ad-hoc developer workflows.

The provider is a thin layer on top of ``subprocess`` that:

1. Detects user intent from natural-language prompts via regex
   (:func:`detect_gh_intent`). Phase 1 covers the most common verbs;
   Phase 2 will route ambiguous prompts to an LLM dispatcher.
2. Runs ``gh`` with the constructed argument list, streaming stdout back
   as :class:`AgentChunk` instances.
3. Reports authentication state via ``gh auth status``.

Design-by-Contract:
    Preconditions:
        - :meth:`GitHubCliProvider.send` requires :meth:`is_available` to
          be ``True``.
        - :meth:`GitHubCliProvider.send_message` requires a non-empty
          message.
    Postconditions:
        - :meth:`cancel` is idempotent — repeated calls are safe even if no
          process is running.
        - :meth:`stream` always yields a final chunk with ``is_final=True``.

Example::

    >>> provider = GitHubCliProvider()
    >>> if provider.is_available():
    ...     result = provider.send("list my issues")
    ...     print(result.stdout)
"""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.shared.python.ai.adapters.base import BaseAgentAdapter, ToolDeclaration
from src.shared.python.ai.types import (
    AgentChunk,
    AgentResponse,
    ConversationContext,
    ProviderCapabilities,
    ProviderCapability,
)
from src.shared.python.contracts import precondition
from src.shared.python.logging_pkg.logging_config import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator

    from src.shared.python.ai.types import ChatModelInfo

logger = get_logger(__name__)

GH_EXECUTABLE = "gh"
DEFAULT_TIMEOUT = 120.0  # [s]


# ---------------------------------------------------------------------------
# Intent detection (Phase 1 — regex)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GhIntent:
    """Resolved ``gh`` invocation derived from a natural-language prompt.

    Attributes:
        args: ``gh`` subcommand arguments (no leading ``gh``).
        requires_confirmation: Destructive intents (create, merge, close,
            delete) carry ``True`` so the chat layer can prompt the user.
    """

    args: list[str] = field(default_factory=list)
    requires_confirmation: bool = False


_TITLE_QUOTED = re.compile(r"""['"]([^'"]+)['"]""")
_REPO_REF = re.compile(r"\b([A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*)\b")
_HASH_NUMBER = re.compile(r"#?(\d+)\b")


def _extract_number(message: str) -> str | None:
    match = _HASH_NUMBER.search(message)
    return match.group(1) if match else None


def _extract_title(message: str) -> str | None:
    match = _TITLE_QUOTED.search(message)
    return match.group(1) if match else None


def _extract_repo(message: str) -> str | None:
    match = _REPO_REF.search(message)
    return match.group(1) if match else None


def detect_gh_intent(message: str) -> GhIntent | None:
    """Map a chat-style message to a ``gh`` argument list.

    Returns ``None`` when no intent can be confidently recognized so the
    caller can surface a help message rather than running a wrong command.

    Args:
        message: User chat message.

    Returns:
        A :class:`GhIntent` describing the ``gh`` subcommand args, or
        ``None`` if no rule matched.
    """
    if not message or not message.strip():
        return None

    text = message.strip().lower()

    # ── issues ────────────────────────────────────────────────────────────
    if re.search(r"\bcreate (an? )?issue\b", text):
        title = _extract_title(message)
        args = ["issue", "create"]
        if title:
            args += ["--title", title]
        return GhIntent(args=args, requires_confirmation=True)

    if re.search(r"\b(close|closing) (an? )?issue\b", text):
        number = _extract_number(message)
        args = ["issue", "close"]
        if number:
            args.append(number)
        return GhIntent(args=args, requires_confirmation=True)

    if re.search(r"\bview (an? )?issue\b", text):
        number = _extract_number(message)
        args = ["issue", "view"]
        if number:
            args.append(number)
        return GhIntent(args=args)

    if re.search(r"\b(list|show)( all| my)? issues?\b", text) or text == "issues":
        args = ["issue", "list"]
        if re.search(r"\bmy\b", text):
            args.append("--me")
        return GhIntent(args=args)

    # ── pull requests ─────────────────────────────────────────────────────
    if re.search(r"\bmerge (a |the )?pr\b", text) or re.search(
        r"\bmerge pull request\b", text
    ):
        number = _extract_number(message)
        args = ["pr", "merge"]
        if number:
            args.append(number)
        return GhIntent(args=args, requires_confirmation=True)

    if re.search(r"\bcreate (a )?pr\b", text) or re.search(
        r"\bcreate pull request\b", text
    ):
        title = _extract_title(message)
        args = ["pr", "create"]
        if title:
            args += ["--title", title]
        return GhIntent(args=args, requires_confirmation=True)

    if re.search(r"\bview (a |the )?pr\b", text) or re.search(
        r"\bview pull request\b", text
    ):
        number = _extract_number(message)
        args = ["pr", "view"]
        if number:
            args.append(number)
        return GhIntent(args=args)

    if re.search(r"\b(list|show)( all| my)? prs?\b", text) or re.search(
        r"\b(list|show)( all| my)? pull requests?\b", text
    ):
        args = ["pr", "list"]
        if re.search(r"\bmy\b", text):
            args.append("--author")
            args.append("@me")
        return GhIntent(args=args)

    # ── repositories ──────────────────────────────────────────────────────
    if re.search(r"\bview (a |the )?repo(sitory)?\b", text):
        repo = _extract_repo(message)
        args = ["repo", "view"]
        if repo:
            args.append(repo)
        return GhIntent(args=args)

    if re.search(r"\b(list|show)( all| my)? repos?(itories)?\b", text):
        return GhIntent(args=["repo", "list"])

    # ── workflow runs ─────────────────────────────────────────────────────
    if re.search(r"\b(list|show)( all)? (workflow )?runs?\b", text):
        return GhIntent(args=["run", "list"])

    # ── catch-all: nothing matched ────────────────────────────────────────
    return None


SUPPORTED_INTENTS_HELP = (
    "I can run gh commands. Supported intents:\n"
    "  - list issues / list my issues / view issue #N\n"
    '  - create issue titled "..." / close issue #N\n'
    '  - list PRs / view PR #N / create PR titled "..." / merge PR #N\n'
    "  - list repos / view repo OWNER/NAME\n"
    "  - list runs\n"
)


# ---------------------------------------------------------------------------
# Result + Provider
# ---------------------------------------------------------------------------


@dataclass
class GitHubCliResult:
    """Captured output of a single ``gh`` invocation.

    Attributes:
        stdout: Captured standard output.
        stderr: Captured standard error.
        exit_code: ``gh`` exit status (``0`` on success).
        args: The arguments passed to ``gh`` (no leading ``gh``).
    """

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    args: list[str] = field(default_factory=list)


class GitHubCliProvider(BaseAgentAdapter):
    """Wrap ``gh`` as a chat-level CLI agent provider.

    Attributes:
        timeout: Per-invocation timeout [s].
    """

    _STATIC_MODELS: tuple[str, ...] = ("gh",)

    def __init__(self, timeout: float | None = None) -> None:
        """Initialize provider.

        Args:
            timeout: Per-invocation timeout [s]. Defaults to 120.
        """
        self._timeout = timeout if timeout is not None else DEFAULT_TIMEOUT
        self._active_proc: subprocess.Popen[str] | None = None
        logger.info("Initialized GitHubCliProvider (timeout=%.1fs)", self._timeout)

    # ------------------------------------------------------------------
    # Availability / auth
    # ------------------------------------------------------------------

    def _gh_path(self) -> str | None:
        """Return resolved ``gh`` path or ``None`` if missing on PATH."""
        return shutil.which(GH_EXECUTABLE)

    def _run_auth_status(self) -> subprocess.CompletedProcess[str]:
        """Run ``gh auth status`` (captured, never raises on non-zero)."""
        return subprocess.run(  # noqa: S603 - args are constants
            [GH_EXECUTABLE, "auth", "status"],
            capture_output=True,
            text=True,
            timeout=self._timeout,
            check=False,
        )

    def is_available(self) -> bool:
        """Return ``True`` when ``gh`` is on PATH and authenticated.

        Probes ``gh auth status`` and considers the provider available iff
        the exit code is 0.
        """
        if self._gh_path() is None:
            return False
        try:
            result = self._run_auth_status()
        except (OSError, subprocess.SubprocessError) as exc:
            logger.debug("gh auth status raised: %s", exc)
            return False
        return result.returncode == 0

    def validate_connection(self) -> tuple[bool, str]:
        """Test ``gh`` availability and authentication state.

        Returns:
            Tuple of (success, diagnostic_message). On failure the
            message includes the recommended ``gh auth login`` hint.
        """
        if self._gh_path() is None:
            return False, (
                "gh CLI not found on PATH. Install from https://cli.github.com/ "
                "to use the GitHub CLI provider."
            )
        try:
            result = self._run_auth_status()
        except (OSError, subprocess.SubprocessError) as exc:
            return False, f"gh auth status failed: {exc}"
        if result.returncode == 0:
            # ``gh auth status`` prints diagnostics to stderr on some
            # platforms even on success; surface whichever stream has text.
            diag = (result.stdout or result.stderr or "Authenticated").strip()
            return True, diag
        msg = (result.stderr or result.stdout or "Not authenticated").strip()
        if "auth login" not in msg.lower():
            msg += "\nHint: run `gh auth login` to authenticate."
        return False, msg

    # ------------------------------------------------------------------
    # Core send / stream (chat-level)
    # ------------------------------------------------------------------

    @precondition(
        lambda self, message: bool(message and message.strip()),
        "message must not be empty or blank",
    )
    def send(self, message: str) -> GitHubCliResult:
        """Translate a chat message into a single ``gh`` invocation.

        Precondition: :meth:`is_available` must be ``True``.

        Args:
            message: Natural-language user message.

        Returns:
            :class:`GitHubCliResult` capturing stdout/stderr/exit_code. If
            no intent is recognized, exit_code is non-zero and stderr
            contains the supported-intent help text.
        """
        if not self.is_available():
            raise RuntimeError(
                "GitHubCliProvider not available — gh missing or not authenticated."
            )

        intent = detect_gh_intent(message)
        if intent is None:
            return GitHubCliResult(
                stdout="",
                stderr=SUPPORTED_INTENTS_HELP,
                exit_code=2,
                args=[],
            )

        args = [GH_EXECUTABLE, *intent.args]
        try:
            cp = subprocess.run(  # noqa: S603 - args constructed from validated intent
                args,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.error("gh invocation failed: %s", exc)
            return GitHubCliResult(
                stdout="",
                stderr=f"gh invocation failed: {exc}",
                exit_code=1,
                args=intent.args,
            )
        return GitHubCliResult(
            stdout=cp.stdout or "",
            stderr=cp.stderr or "",
            exit_code=cp.returncode,
            args=intent.args,
        )

    def stream(self, message: str) -> Iterator[AgentChunk]:
        """Stream ``gh`` stdout lines as :class:`AgentChunk` instances.

        Always emits a final chunk with ``is_final=True``. If no intent is
        recognized the help text is yielded as a single final chunk.

        Args:
            message: Natural-language user message.

        Yields:
            :class:`AgentChunk` per stdout line (final chunk closes
            the stream).
        """
        if not self.is_available():
            yield AgentChunk(
                content=(
                    "GitHubCliProvider not available — install gh and run "
                    "`gh auth login`."
                ),
                is_final=True,
                index=0,
            )
            return

        intent = detect_gh_intent(message)
        if intent is None:
            yield AgentChunk(
                content=SUPPORTED_INTENTS_HELP,
                is_final=True,
                index=0,
            )
            return

        args = [GH_EXECUTABLE, *intent.args]
        index = 0
        try:
            proc = subprocess.Popen(  # noqa: S603
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            yield AgentChunk(
                content=f"Failed to spawn gh: {exc}",
                is_final=True,
                index=index,
            )
            return

        self._active_proc = proc
        try:
            if proc.stdout is not None:
                for line in proc.stdout:
                    yield AgentChunk(content=line, is_final=False, index=index)
                    index += 1
            proc.wait()
        finally:
            self._active_proc = None

        yield AgentChunk(content="", is_final=True, index=index)

    def cancel(self) -> None:
        """Terminate any in-flight ``gh`` process. Idempotent.

        Safe to call when no process is running.
        """
        proc = self._active_proc
        if proc is None:
            return
        try:
            proc.terminate()
        except (OSError, subprocess.SubprocessError) as exc:
            logger.debug("cancel: terminate raised %s", exc)
        finally:
            self._active_proc = None

    # ------------------------------------------------------------------
    # BaseAgentAdapter surface
    # ------------------------------------------------------------------

    @precondition(
        lambda self, message, context, tools: bool(message and message.strip()),
        "message must not be empty or blank",
    )
    def send_message(
        self,
        message: str,
        context: ConversationContext,
        tools: list[ToolDeclaration],
    ) -> AgentResponse:
        """Synchronous send returning an :class:`AgentResponse`.

        The ``tools`` argument is accepted for protocol compliance but is
        unused: ``gh`` itself defines the tool surface.
        """
        del tools, context  # parameters unused; see docstring
        result = self.send(message)
        content_parts: list[str] = []
        if result.args:
            cmd = [GH_EXECUTABLE, *result.args]
            content_parts.append("$ " + " ".join(shlex.quote(a) for a in cmd))
        if result.stdout:
            content_parts.append(result.stdout.rstrip())
        if result.stderr:
            content_parts.append(result.stderr.rstrip())
        content = "\n".join(p for p in content_parts if p)
        finish = "stop" if result.exit_code == 0 else "error"
        return AgentResponse(
            content=content,
            finish_reason=finish,
            metadata={"exit_code": result.exit_code, "args": result.args},
        )

    def stream_response(
        self,
        message: str,
        context: ConversationContext,
        tools: list[ToolDeclaration],
    ) -> Iterator[AgentChunk]:
        """Stream ``gh`` output as :class:`AgentChunk` instances.

        Mirrors :meth:`stream`; ``context`` and ``tools`` are unused.
        """
        del tools, context
        yield from self.stream(message)

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Return provider capabilities."""
        return ProviderCapabilities(
            supported=frozenset({ProviderCapability.STREAMING}),
            max_tokens=0,  # not token-based
            model_name=GH_EXECUTABLE,
            provider_name="github-cli",
        )

    def list_models(self) -> list[ChatModelInfo]:
        """``gh`` exposes a single virtual "model"."""
        from src.shared.python.ai.types import ChatModelInfo

        return [
            ChatModelInfo(
                name=m,
                provider="github-cli",
                display_name=m,
                id=m,
            )
            for m in self._STATIC_MODELS
        ]

    def thinking_capabilities(self) -> Any:
        """``gh`` does not reason — expose only the ``none`` level."""
        from src.shared.python.chat.models import make_none_only_capabilities

        return make_none_only_capabilities(provider="github-cli")
