"""Project-scoped runtime for terminal-backed chat agent sessions."""

from __future__ import annotations

import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .terminal_contracts import (
    TerminalAgentEvent,
    TerminalAgentSessionInfo,
    TerminalAgentSessionRequest,
    TerminalProviderRegistry,
)

_POWERSHELL_IDS = {"powershell", "pwsh"}
_BASH_IDS = {"bash", "wsl"}

# ---------------------------------------------------------------------------
# Minimal environment allowlist for spawned agent sessions.
#
# When no explicit base_env is provided, the runtime builds a sanitised copy
# of os.environ that contains only well-known, non-sensitive variables.
# Any variable whose name matches a credential-like pattern is excluded so
# that parent-process API keys and secrets never leak into agent subprocesses.
# Callers that need specific credentials must pass them explicitly via
# base_env; that signal makes the intent deliberate and reviewable.
# ---------------------------------------------------------------------------

_SESSION_ENV_ALLOWLIST: frozenset[str] = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "USERNAME",
        "USERPROFILE",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LANGUAGE",
        "TZ",
        "TERM",
        "TEMP",
        "TMP",
        "TMPDIR",
        # Windows minimums
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        # POSIX minimums
        "SHELL",
    }
)

_SESSION_ENV_SENSITIVE_PATTERNS: tuple[str, ...] = (
    "_API_KEY",
    "_TOKEN",
    "_SECRET",
    "_PASSWORD",
    "_CREDENTIAL",
    "_PRIVATE_KEY",
    "_ACCESS_KEY",
)


def _is_sensitive(name: str) -> bool:
    """Return True when *name* looks like a credential variable."""
    upper = name.upper()
    return any(pat in upper for pat in _SESSION_ENV_SENSITIVE_PATTERNS)


def _build_default_session_env() -> dict[str, str]:
    """Build a minimal environment mapping for a spawned agent session.

    Includes only variables on the ``_SESSION_ENV_ALLOWLIST`` (PATH, HOME,
    TZ and similar OS-level vars) and excludes anything whose name matches a
    credential-like pattern (``*_API_KEY``, ``*_TOKEN``, ``*_SECRET``,
    ``*_PASSWORD``, ``*_CREDENTIAL``, ``*_PRIVATE_KEY``, ``*_ACCESS_KEY``).

    Callers that need specific environment variables — including API keys —
    must pass them explicitly via the ``base_env`` argument on
    ``TerminalSessionRuntime``.  That makes the intent deliberate and
    auditable rather than relying on implicit inheritance from the parent
    process.
    """
    return {
        k: v
        for k, v in os.environ.items()
        if k in _SESSION_ENV_ALLOWLIST and not _is_sensitive(k)
    }


class TerminalRuntimeError(RuntimeError):
    """Raised when a terminal session lifecycle operation is invalid."""


@dataclass(frozen=True)
class ProcessLaunchRequest:
    """Provider process launch request built by the terminal runtime."""

    session_id: str
    command: list[str]
    cwd: Path
    env: dict[str, str]


class TerminalProcessAdapter(Protocol):
    """Process boundary used by terminal runtime implementations."""

    def start(self, request: ProcessLaunchRequest) -> str:
        """Start a process and return an adapter-specific process id."""

    def write(self, process_id: str, text: str) -> None:
        """Write user input to the process."""

    def resize(self, process_id: str, columns: int, rows: int) -> None:
        """Resize the process terminal surface."""

    def stop(self, process_id: str) -> None:
        """Stop the process and release resources."""

    def drain_events(self, process_id: str) -> list[TerminalAgentEvent]:
        """Return available normalized process events."""


@dataclass
class _RuntimeSession:
    info: TerminalAgentSessionInfo
    process_id: str


class TerminalSessionRuntime:
    """Orchestrates project-scoped terminal-agent process sessions."""

    def __init__(
        self,
        registry: TerminalProviderRegistry,
        process_adapter: TerminalProcessAdapter,
        *,
        allowed_roots: Sequence[Path] | None = None,
        base_env: Mapping[str, str] | None = None,
    ) -> None:
        self._registry = registry
        self._process_adapter = process_adapter
        self._allowed_roots = [
            root.expanduser().resolve() for root in (allowed_roots or [])
        ]
        self._base_env = (
            dict(base_env) if base_env is not None else _build_default_session_env()
        )
        self._sessions: dict[str, _RuntimeSession] = {}

    def start(
        self,
        request: TerminalAgentSessionRequest,
    ) -> TerminalAgentSessionInfo:
        """Start a terminal-agent session in the requested project root."""
        self._validate_project_root(request.project_root)
        selection = self._registry.validate_selection(
            request.shell_id,
            request.provider_id,
        )
        session_id = request.session_id or f"terminal_{uuid.uuid4().hex[:12]}"
        command = _build_command(
            shell_id=selection.shell.id,
            shell_executable=selection.shell.executable,
            shell_args=selection.shell.default_args,
            provider_executable=selection.provider.executable,
            provider_args=[
                *selection.provider.launch_args,
                *request.provider_args,
            ],
        )
        env = self._session_env(request, session_id)
        launch = ProcessLaunchRequest(
            session_id=session_id,
            command=command,
            cwd=request.project_root,
            env=env,
        )
        process_id = self._process_adapter.start(launch)
        info = TerminalAgentSessionInfo(
            session_id=session_id,
            resolved_cwd=request.project_root,
            shell_id=selection.shell.id,
            provider_id=selection.provider.id,
            state="running",
            diagnostics={"process_id": process_id},
        )
        self._sessions[session_id] = _RuntimeSession(info=info, process_id=process_id)
        return info

    def get_session(self, session_id: str) -> TerminalAgentSessionInfo:
        """Return current state for a terminal session."""
        return self._session_for(session_id).info

    def write(self, session_id: str, text: str) -> None:
        """Write terminal input to the session's process."""
        if not text:
            raise TerminalRuntimeError("terminal input must be non-empty")
        session = self._session_for(session_id)
        self._process_adapter.write(session.process_id, text)

    def resize(self, session_id: str, *, columns: int, rows: int) -> None:
        """Resize the session's terminal surface."""
        if columns <= 0 or rows <= 0:
            raise TerminalRuntimeError("terminal size must be positive")
        session = self._session_for(session_id)
        self._process_adapter.resize(session.process_id, columns, rows)

    def stop(self, session_id: str) -> TerminalAgentSessionInfo:
        """Stop the session's process and mark it stopped."""
        session = self._session_for(session_id)
        self._process_adapter.stop(session.process_id)
        session.info = session.info.model_copy(update={"state": "stopped"})
        return session.info

    def drain_events(self, session_id: str) -> list[TerminalAgentEvent]:
        """Drain normalized terminal events for a session."""
        session = self._session_for(session_id)
        return self._process_adapter.drain_events(session.process_id)

    def _validate_project_root(self, project_root: Path) -> None:
        if not self._allowed_roots:
            return
        if not any(_is_relative_to(project_root, root) for root in self._allowed_roots):
            raise TerminalRuntimeError(
                f"project root {project_root} is not under an allowed root"
            )

    def _session_env(
        self,
        request: TerminalAgentSessionRequest,
        session_id: str,
    ) -> dict[str, str]:
        env = dict(self._base_env)
        env.update(
            {
                "TOOLS_CHAT_APP_CONTEXT": request.app_context,
                "TOOLS_CHAT_PROJECT_ROOT": str(request.project_root),
                "TOOLS_CHAT_SESSION_ID": session_id,
            }
        )
        return env

    def _session_for(self, session_id: str) -> _RuntimeSession:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise TerminalRuntimeError(
                f"unknown terminal session {session_id!r}"
            ) from exc


def _build_command(
    *,
    shell_id: str,
    shell_executable: str,
    shell_args: Sequence[str],
    provider_executable: str,
    provider_args: Sequence[str],
) -> list[str]:
    provider_command = [provider_executable, *provider_args]
    if shell_id in _POWERSHELL_IDS:
        return [shell_executable, *shell_args, "-Command", *provider_command]
    if shell_id in _BASH_IDS:
        return [shell_executable, *shell_args, "-lc", " ".join(provider_command)]
    return [shell_executable, *shell_args, *provider_command]


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
