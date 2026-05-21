"""CLI-provider installation + authentication guide.

Single source of truth for "is this CLI installed", "if not, what's the
install command", and "how does the user authenticate" — for every chat
provider that wraps a CLI binary. The chat UI uses this to render
get-started cards when a user picks a provider whose CLI is missing.

The data here intentionally does NOT run the install commands itself —
launching package managers and opening browser flows on the user's behalf
is the chat dock's responsibility (it knows how to open a terminal with the
right cwd). This module just exposes a structured catalogue so the UI does
not have to hardcode install URLs and command strings.

Example::

    >>> from src.shared.python.ai.adapters.cli_provider_setup import (
    ...     get_cli_setup_status,
    ...     CLI_PROVIDERS,
    ... )
    >>> status = get_cli_setup_status("claude_code")
    >>> if not status.installed:
    ...     print(status.install_command)
    ...     print(status.auth_instructions)

Design-by-contract:
    - :func:`get_cli_setup_status` always returns a :class:`CliSetupStatus`
      (never raises) so the chat UI can call it during widget construction.
    - :data:`CLI_PROVIDERS` is the closed enumeration of CLI-shaped
      providers; new entries must include both install and auth commands.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class CliSetupStatus:
    """Result of an install/auth probe for one CLI provider.

    Attributes:
        provider: Provider identifier (matches ``AdapterFactory`` keys).
        display_name: Human-readable name for UI labels.
        installed: ``True`` when the CLI binary is on PATH or in a known
            install location.
        binary_path: Resolved absolute path of the binary, or ``None``
            when missing.
        install_command: Shell command the user should run to install the
            CLI. Always non-empty; surface in a copy-to-clipboard widget.
        install_url: Authoritative docs URL with manual install instructions.
        auth_instructions: Multi-line, human-readable description of how to
            authenticate. May tell the user to "run ``codex login``" or
            "set ``ANTHROPIC_API_KEY``" depending on what the CLI supports.
        auth_command: Optional one-line command that opens the auth flow.
            ``None`` when no headless command exists (e.g. interactive
            OAuth flows).
        version: CLI version string when ``installed`` (best-effort).
    """

    provider: str
    display_name: str
    installed: bool
    binary_path: str | None
    install_command: str
    install_url: str
    auth_instructions: str
    auth_command: str | None
    version: str | None = None


@dataclass(frozen=True)
class _CliProviderSpec:
    """Static metadata for one CLI-shaped provider."""

    provider: str
    display_name: str
    install_command: str
    install_url: str
    auth_instructions: str
    auth_command: str | None
    # Callable returning (binary_path | None, version | None). Indirected so
    # tests can patch this; concrete adapters supply their own resolvers.
    probe: Callable[[], tuple[str | None, str | None]] = field(repr=False)


def _probe_claude_code() -> tuple[str | None, str | None]:
    from src.shared.python.ai.adapters.claude_code_adapter import _resolve_binary

    binary = _resolve_binary()
    if binary is None:
        return None, None
    return binary, _safe_version(binary, ["--version"])


def _probe_codex_cli() -> tuple[str | None, str | None]:
    from src.shared.python.ai.adapters.codex_cli_adapter import _resolve_binary

    binary = _resolve_binary()
    if binary is None:
        return None, None
    return binary, _safe_version(binary, ["--version"])


def _probe_gemini_cli() -> tuple[str | None, str | None]:
    from src.shared.python.ai.adapters.gemini_cli_adapter import _resolve_binary

    binary = _resolve_binary()
    if binary is None:
        return None, None
    return binary, _safe_version(binary, ["--version"])


def _probe_cline() -> tuple[str | None, str | None]:
    # Cline is currently shipped as a VS Code extension, not a CLI. We
    # still expose a setup card so the user sees "install the Cline
    # extension in VS Code" instead of silent missing-provider state.
    import shutil

    binary = shutil.which("cline")
    if binary is None:
        return None, None
    return binary, _safe_version(binary, ["--version"])


def _safe_version(binary: str, args: list[str]) -> str | None:
    """Run ``binary <args>`` with a tight timeout; never raise."""
    import subprocess

    try:
        result = subprocess.run(
            [binary, *args],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
            encoding="utf-8",
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    lines = (result.stdout or "").strip().splitlines()
    return lines[0] if lines else None


CLI_PROVIDERS: dict[str, _CliProviderSpec] = {
    "claude_code": _CliProviderSpec(
        provider="claude_code",
        display_name="Claude Code",
        install_command="curl -fsSL https://claude.ai/install.sh | bash",
        install_url="https://docs.claude.com/en/docs/claude-code/quickstart",
        auth_instructions=(
            "Run `claude` once in a terminal. The CLI will open a browser "
            "for OAuth login and store the credential in the OS keychain. "
            "Alternatively, set the ANTHROPIC_API_KEY environment variable "
            "before launching the app."
        ),
        auth_command="claude login",
        probe=_probe_claude_code,
    ),
    "codex_cli": _CliProviderSpec(
        provider="codex_cli",
        display_name="OpenAI Codex CLI",
        install_command="npm install -g @openai/codex",
        install_url="https://github.com/openai/codex",
        auth_instructions=(
            "Run `codex login` in a terminal to authenticate via OpenAI's "
            "device-code flow. Alternatively, export OPENAI_API_KEY in the "
            "environment before launching the app."
        ),
        auth_command="codex login",
        probe=_probe_codex_cli,
    ),
    "gemini_cli": _CliProviderSpec(
        provider="gemini_cli",
        display_name="Google Gemini CLI",
        install_command="npm install -g @google/gemini-cli",
        install_url="https://github.com/google-gemini/gemini-cli",
        auth_instructions=(
            "Run `gemini` once in a terminal. The CLI walks through an "
            "OAuth login (Google account) or you can set GEMINI_API_KEY "
            "from https://aistudio.google.com/apikey before launching."
        ),
        auth_command="gemini",
        probe=_probe_gemini_cli,
    ),
    "cline": _CliProviderSpec(
        provider="cline",
        display_name="Cline",
        # Cline is primarily a VS Code extension; the install command opens
        # the extension marketplace. Surfacing the marketplace URL is the
        # correct guidance until there is an official Cline CLI.
        install_command="code --install-extension saoudrizwan.claude-dev",
        install_url="https://cline.bot/",
        auth_instructions=(
            "Cline runs as a VS Code extension. Install it (the command "
            "above runs from any terminal with VS Code on PATH), then "
            "configure the underlying provider (Claude/OpenAI/etc.) inside "
            "the Cline panel. The local Cline HTTP server (default port "
            "3000) is what this chat connects to."
        ),
        auth_command=None,
        probe=_probe_cline,
    ),
}


def get_cli_setup_status(provider: str) -> CliSetupStatus:
    """Return install + auth status for one CLI provider.

    Args:
        provider: One of the keys of :data:`CLI_PROVIDERS`.

    Returns:
        Populated :class:`CliSetupStatus`. Never raises — unknown providers
        return ``installed=False`` with empty fields so the caller can
        safely render a fallback card.
    """
    spec = CLI_PROVIDERS.get(provider)
    if spec is None:
        return CliSetupStatus(
            provider=provider,
            display_name=provider,
            installed=False,
            binary_path=None,
            install_command="",
            install_url="",
            auth_instructions=(
                f"Unknown CLI provider '{provider}'. See the chat dock "
                "documentation for the supported provider list."
            ),
            auth_command=None,
            version=None,
        )
    try:
        binary, version = spec.probe()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "CLI probe for %s raised %s: %s — treating as not installed",
            provider,
            type(exc).__name__,
            exc,
        )
        binary, version = None, None
    return CliSetupStatus(
        provider=spec.provider,
        display_name=spec.display_name,
        installed=binary is not None,
        binary_path=binary,
        install_command=spec.install_command,
        install_url=spec.install_url,
        auth_instructions=spec.auth_instructions,
        auth_command=spec.auth_command,
        version=version,
    )


def get_all_cli_setup_statuses() -> dict[str, CliSetupStatus]:
    """Return the install + auth status for every CLI provider.

    Use this to populate a "manage providers" page; for a single provider
    use :func:`get_cli_setup_status` instead.
    """
    return {p: get_cli_setup_status(p) for p in CLI_PROVIDERS}
