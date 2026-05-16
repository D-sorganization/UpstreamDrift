"""Discover CLI agent providers installed on the user's machine.

Combines ``shutil.which`` lookups with config-directory probes so
known providers are found even when their executable is shadowed or
when the user opted into a non-default install location.

Only providers whose primary indicator actually exists are returned —
the chat header dropdown shows the result verbatim, so callers can
rely on every returned descriptor being launchable.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from src.shared.python.ai.cli_providers.contracts import CliProviderDescriptor

# (provider_id, display_name, executable_name, config_subpath)
_KNOWN_PROVIDERS: tuple[tuple[str, str, str, str], ...] = (
    ("claude-cli", "Claude CLI", "claude", ".claude"),
    ("codex-cli", "Codex CLI", "codex", ".codex"),
)


def _resolve_executable(name: str) -> str:
    """Return the absolute path to ``name`` on PATH, or empty.

    On Windows, ``shutil.which`` already handles the ``.exe`` suffix,
    so callers pass the bare command name.
    """
    found = shutil.which(name)
    return found or ""


def _has_config_dir(subpath: str) -> bool:
    """Return True if ``~/<subpath>`` exists (and is a directory).

    Used as a fallback signal: a user who has the provider installed
    almost always has a config dir even if the binary moved.
    """
    home = Path(os.path.expanduser("~"))
    candidate = home / subpath
    return candidate.is_dir()


def _discover_stdio_providers() -> list[CliProviderDescriptor]:
    """Return descriptors for stdio-transport providers found locally."""
    discovered: list[CliProviderDescriptor] = []
    for provider_id, name, exe, subpath in _KNOWN_PROVIDERS:
        path = _resolve_executable(exe)
        if not path and not _has_config_dir(subpath):
            continue
        discovered.append(
            CliProviderDescriptor(
                id=provider_id,
                name=name,
                executable_path=path,
                transport="stdio",
                required_env=(),
                working_dir_aware=True,
            )
        )
    return discovered


def _discover_cline() -> CliProviderDescriptor | None:
    """Return a Cline descriptor when its config directory exists.

    The actual socket reachability is checked at send() time so the
    descriptor can still appear in the dropdown when Cline is
    installed but not currently running.
    """
    if not _has_config_dir(".cline") and not _has_config_dir(".vscode"):
        return None
    return CliProviderDescriptor(
        id="cline",
        name="Cline",
        executable_path="",
        transport="socket",
        required_env=(),
        working_dir_aware=False,
    )


def discover_cli_providers() -> list[CliProviderDescriptor]:
    """Discover all CLI agent providers installed on this system.

    Returns:
        List of descriptors, one per installed provider. Empty when
        none are found. Order is stable: Claude CLI, Codex CLI, Cline.

    Contract:
        Post: every returned descriptor has either a non-empty
            ``executable_path`` (stdio) or a transport of ``"socket"``.
    """
    descriptors = _discover_stdio_providers()
    cline = _discover_cline()
    if cline is not None:
        descriptors.append(cline)
    return descriptors
