"""Default terminal-agent provider descriptors for shared chat integrations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, NamedTuple

from .terminal_contracts import (
    TerminalAgentProviderInfo,
    TerminalPathStyle,
    TerminalProviderRegistry,
    TerminalShellInfo,
)

_REDACTED_VALUE: Final = "***"
_SENSITIVE_OPTION_NAMES: Final = {
    "--api-key",
    "--apikey",
    "--auth-token",
    "--client-secret",
    "--password",
    "--secret",
    "--token",
}
_SENSITIVE_KEY_PARTS: Final = (
    "api_key",
    "apikey",
    "auth_token",
    "client_secret",
    "password",
    "secret",
    "token",
)


class _ShellDefinition(NamedTuple):
    id: str
    display_name: str
    executable: str
    default_args: tuple[str, ...] = ()
    platforms: tuple[str, ...] = ()
    path_style: TerminalPathStyle = "native"


class _ProviderDefinition(NamedTuple):
    id: str
    display_name: str
    executable: str
    supported_shells: tuple[str, ...]
    install_probe_args: tuple[str, ...] = ("--version",)
    auth_probe_args: tuple[str, ...] = ()


_SHELL_DEFINITIONS: Final = (
    _ShellDefinition(
        id="powershell",
        display_name="Windows PowerShell",
        executable="powershell",
        default_args=("-NoLogo", "-NoProfile"),
        platforms=("win32",),
    ),
    _ShellDefinition(
        id="pwsh",
        display_name="PowerShell",
        executable="pwsh",
        default_args=("-NoLogo", "-NoProfile"),
    ),
    _ShellDefinition(
        id="bash",
        display_name="Bash",
        executable="bash",
        platforms=("linux", "darwin"),
        path_style="posix",
    ),
    _ShellDefinition(
        id="wsl",
        display_name="WSL",
        executable="wsl",
        platforms=("win32",),
        path_style="wsl",
    ),
)
_SUPPORTED_SHELL_IDS: Final = ("powershell", "pwsh", "bash", "wsl")
_PROVIDER_DEFINITIONS: Final = (
    _ProviderDefinition(
        id="claude-code",
        display_name="Claude Code",
        executable="claude",
        supported_shells=_SUPPORTED_SHELL_IDS,
        auth_probe_args=("auth", "status"),
    ),
    _ProviderDefinition(
        id="codex",
        display_name="Codex",
        executable="codex",
        supported_shells=_SUPPORTED_SHELL_IDS,
        auth_probe_args=("auth", "status"),
    ),
    _ProviderDefinition(
        id="cline-cli",
        display_name="Cline CLI",
        executable="cline",
        supported_shells=_SUPPORTED_SHELL_IDS,
    ),
    _ProviderDefinition(
        id="gemini-cli",
        display_name="Gemini CLI",
        executable="gemini",
        supported_shells=_SUPPORTED_SHELL_IDS,
    ),
)


def default_terminal_shells() -> list[TerminalShellInfo]:
    """Return default shell descriptors for terminal-agent sessions."""
    return [
        TerminalShellInfo(
            id=definition.id,
            display_name=definition.display_name,
            executable=definition.executable,
            default_args=list(definition.default_args),
            platforms=list(definition.platforms),
            path_style=definition.path_style,
        )
        for definition in _SHELL_DEFINITIONS
    ]


def default_terminal_agent_providers() -> list[TerminalAgentProviderInfo]:
    """Return default terminal-agent provider descriptors.

    The descriptors are UI-agnostic metadata only. They do not probe the local
    machine or imply that a provider executable is installed.
    """
    return [
        TerminalAgentProviderInfo(
            id=definition.id,
            display_name=definition.display_name,
            executable=definition.executable,
            supported_shells=list(definition.supported_shells),
            install_probe_args=list(definition.install_probe_args),
            auth_probe_args=list(definition.auth_probe_args),
        )
        for definition in _PROVIDER_DEFINITIONS
    ]


def build_default_terminal_provider_registry() -> TerminalProviderRegistry:
    """Build a registry populated with the default shells and providers."""
    registry = TerminalProviderRegistry()
    for shell in default_terminal_shells():
        registry.register_shell(shell)
    for provider in default_terminal_agent_providers():
        registry.register_provider(provider)
    return registry


def provider_probe_commands(provider_id: str) -> dict[str, list[str]]:
    """Return install/auth probe commands for a default provider id.

    Raises:
        KeyError: If ``provider_id`` is not one of the default providers.
    """
    provider = _default_provider_by_id(provider_id)
    probes = {"install": [provider.executable, *provider.install_probe_args]}
    if provider.auth_probe_args:
        probes["auth"] = [provider.executable, *provider.auth_probe_args]
    return probes


def redact_terminal_command(command: Sequence[str]) -> list[str]:
    """Redact secret-like option values from command diagnostics."""
    redacted: list[str] = []
    redact_next = False
    for argument in command:
        if redact_next:
            redacted.append(_REDACTED_VALUE)
            redact_next = False
            continue
        option_name, separator, _ = argument.partition("=")
        normalized_name = _normalize_option_name(option_name)
        if separator and _is_sensitive_option(normalized_name):
            redacted.append(f"{option_name}={_REDACTED_VALUE}")
            continue
        if _is_sensitive_option(normalized_name):
            redacted.append(argument)
            redact_next = True
            continue
        redacted.append(argument)
    return redacted


def _default_provider_by_id(provider_id: str) -> TerminalAgentProviderInfo:
    for provider in default_terminal_agent_providers():
        if provider.id == provider_id:
            return provider
    raise KeyError(f"unknown default terminal provider {provider_id!r}")


def _normalize_option_name(option_name: str) -> str:
    return option_name.strip().lower().replace("-", "_")


def _is_sensitive_option(option_name: str) -> bool:
    if option_name.replace("_", "-") in _SENSITIVE_OPTION_NAMES:
        return True
    return any(part in option_name for part in _SENSITIVE_KEY_PARTS)
