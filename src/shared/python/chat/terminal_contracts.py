"""Shared contracts for terminal-backed chat agent sessions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

TerminalSessionState = Literal[
    "starting",
    "running",
    "stopped",
    "exited",
    "error",
    "auth_required",
]
TerminalEventType = Literal[
    "stdout",
    "stderr",
    "data",
    "title",
    "exit",
    "error",
    "auth_required",
    "status",
]
TerminalPathStyle = Literal["native", "posix", "wsl"]

_ID_PATTERN = r"^[a-z][a-z0-9_-]*$"


class TerminalRegistryError(ValueError):
    """Raised when terminal shell/provider registry contracts are violated."""


class TerminalShellInfo(BaseModel):
    """Descriptor for a supported shell runtime.

    Preconditions:
        - ``id`` is a stable lowercase registry id.
        - ``executable`` is the command used to probe or launch the shell.
    """

    id: str = Field(..., pattern=_ID_PATTERN)
    display_name: str = Field(..., min_length=1)
    executable: str = Field(..., min_length=1)
    default_args: list[str] = Field(default_factory=list)
    platforms: list[str] = Field(default_factory=list)
    path_style: TerminalPathStyle = "native"


class TerminalAgentProviderInfo(BaseModel):
    """Descriptor for a terminal-backed agent provider."""

    id: str = Field(..., pattern=_ID_PATTERN)
    display_name: str = Field(..., min_length=1)
    executable: str = Field(..., min_length=1)
    supported_shells: list[str] = Field(..., min_length=1)
    install_probe_args: list[str] = Field(default_factory=lambda: ["--version"])
    auth_probe_args: list[str] = Field(default_factory=list)
    launch_args: list[str] = Field(default_factory=list)
    setup_args: list[str] = Field(default_factory=list)
    supports_interactive: bool = True
    supports_json_mode: bool = False
    required_env_keys: list[str] = Field(default_factory=list)

    @field_validator("supported_shells")
    @classmethod
    def _validate_supported_shells(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("supported_shells must not contain duplicates")
        for shell_id in value:
            if not shell_id:
                raise ValueError("supported_shells must contain non-empty ids")
            if shell_id.lower() != shell_id:
                raise ValueError("supported_shell ids must be lowercase")
        return value


class TerminalAgentSessionRequest(BaseModel):
    """Request to start a project-scoped terminal-agent session."""

    app_context: str = Field(..., min_length=1)
    project_root: Path
    shell_id: str = Field(..., pattern=_ID_PATTERN)
    provider_id: str = Field(..., pattern=_ID_PATTERN)
    session_id: str | None = None
    provider_args: list[str] = Field(default_factory=list)

    @field_validator("project_root")
    @classmethod
    def _resolve_project_root(cls, value: Path) -> Path:
        resolved = value.expanduser().resolve()
        if not resolved.exists() or not resolved.is_dir():
            raise ValueError("project_root must resolve to an existing directory")
        return resolved


class TerminalAgentSessionInfo(BaseModel):
    """Lifecycle state for a terminal-agent session."""

    model_config = ConfigDict(frozen=True)

    session_id: str = Field(..., min_length=1)
    resolved_cwd: Path
    shell_id: str = Field(..., pattern=_ID_PATTERN)
    provider_id: str = Field(..., pattern=_ID_PATTERN)
    state: TerminalSessionState
    diagnostics: dict[str, str] = Field(default_factory=dict)


class TerminalAgentEvent(BaseModel):
    """Normalized event emitted by a terminal-agent session."""

    session_id: str = Field(..., min_length=1)
    event_type: TerminalEventType
    sequence: int = Field(0, ge=0)
    data: str = ""
    exit_code: int | None = None


@dataclass(frozen=True)
class TerminalSelection:
    """Resolved shell/provider pair returned by registry validation."""

    shell: TerminalShellInfo
    provider: TerminalAgentProviderInfo


class TerminalProviderRegistry:
    """In-memory registry for terminal shells and agent providers."""

    def __init__(self) -> None:
        self._shells: dict[str, TerminalShellInfo] = {}
        self._providers: dict[str, TerminalAgentProviderInfo] = {}

    def register_shell(self, shell: TerminalShellInfo) -> None:
        """Register a shell descriptor.

        Raises:
            TerminalRegistryError: If ``shell.id`` is already registered.
        """
        if shell.id in self._shells:
            raise TerminalRegistryError(f"shell {shell.id!r} is already registered")
        self._shells[shell.id] = shell

    def register_provider(self, provider: TerminalAgentProviderInfo) -> None:
        """Register a provider descriptor after shell compatibility checks."""
        if provider.id in self._providers:
            raise TerminalRegistryError(
                f"provider {provider.id!r} is already registered"
            )
        for shell_id in provider.supported_shells:
            if shell_id not in self._shells:
                raise TerminalRegistryError(
                    f"provider {provider.id!r} references unknown shell {shell_id!r}"
                )
        self._providers[provider.id] = provider

    def get_shell(self, shell_id: str) -> TerminalShellInfo:
        """Return a registered shell descriptor."""
        try:
            return self._shells[shell_id]
        except KeyError as exc:
            raise TerminalRegistryError(f"unknown shell {shell_id!r}") from exc

    def get_provider(self, provider_id: str) -> TerminalAgentProviderInfo:
        """Return a registered provider descriptor."""
        try:
            return self._providers[provider_id]
        except KeyError as exc:
            raise TerminalRegistryError(f"unknown provider {provider_id!r}") from exc

    def shells(self) -> list[TerminalShellInfo]:
        """Return shell descriptors in registration order."""
        return list(self._shells.values())

    def providers(self) -> list[TerminalAgentProviderInfo]:
        """Return provider descriptors in registration order."""
        return list(self._providers.values())

    def providers_for_shell(self, shell_id: str) -> list[TerminalAgentProviderInfo]:
        """Return providers compatible with ``shell_id``."""
        self.get_shell(shell_id)
        return [
            provider
            for provider in self._providers.values()
            if shell_id in provider.supported_shells
        ]

    def validate_selection(
        self,
        shell_id: str,
        provider_id: str,
    ) -> TerminalSelection:
        """Resolve and validate a shell/provider selection."""
        shell = self.get_shell(shell_id)
        provider = self.get_provider(provider_id)
        if shell.id not in provider.supported_shells:
            raise TerminalRegistryError(
                f"provider {provider.id!r} does not support shell {shell.id!r}"
            )
        return TerminalSelection(shell=shell, provider=provider)
