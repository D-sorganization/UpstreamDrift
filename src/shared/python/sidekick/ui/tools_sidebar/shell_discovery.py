"""Shell discovery helpers for the Sidekick OS Terminal (UpstreamDrift #5617).

This module is the **single source of truth** (DRY) for which interactive
shells the Sidekick OS Terminal can launch:

* :class:`ShellDescriptor` — immutable record describing one shell.
* :func:`discover_shells` — returns the user's available shells.

The widget, settings dropdown, and tests all consume :func:`discover_shells`;
no caller must hard-code shell paths.
"""

from __future__ import annotations

import logging
import platform
import shutil
import subprocess
from dataclasses import dataclass

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShellDescriptor:
    """One discovered shell, addressable by ``identifier``.

    Attributes:
        identifier: Stable id used by callers (e.g. ``bash``, ``pwsh``,
            ``wsl:Ubuntu-22.04``).
        label: Human-readable name shown in the shell selector.
        command: Argument vector passed to :func:`subprocess.Popen`
            (or to the PTY backend). Must contain at least one element.

    Raises:
        ValueError: If ``identifier`` or ``label`` is empty, or
            ``command`` is empty (DbC precondition).
    """

    identifier: str
    label: str
    command: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.identifier:
            raise ValueError("identifier must be a non-empty str")
        if not self.label:
            raise ValueError("label must be a non-empty str")
        if not self.command:
            raise ValueError("command must contain at least one element")


# Candidate POSIX shells, in preference order.
_POSIX_CANDIDATES: tuple[str, ...] = ("bash", "zsh", "fish", "sh")

# Candidate Windows shells, in preference order.
_WINDOWS_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("powershell", "Windows PowerShell"),
    ("pwsh", "PowerShell 7"),
    ("cmd", "Command Prompt"),
)


def discover_shells() -> list[ShellDescriptor]:
    """Return the interactive shells available on this host.

    The result is the concatenation of:

    * POSIX shells found on ``PATH`` (Linux, macOS).
    * Windows shells (Windows only).
    * WSL distributions enumerated via ``wsl --list --quiet`` (Windows only,
      when the ``wsl`` executable is present).

    The function never raises; missing shells simply produce an empty list.
    """
    system = platform.system()
    shells: list[ShellDescriptor] = []
    if system in {"Linux", "Darwin"}:
        shells.extend(_discover_posix_shells())
    elif system == "Windows":
        shells.extend(_discover_windows_shells())
        shells.extend(_discover_wsl_distros())
    return shells


def _discover_posix_shells() -> list[ShellDescriptor]:
    """Return POSIX shells available on the current ``PATH``."""
    found: list[ShellDescriptor] = []
    for name in _POSIX_CANDIDATES:
        path = shutil.which(name)
        if path is None:
            continue
        found.append(
            ShellDescriptor(identifier=name, label=name, command=(path,)),
        )
    return found


def _discover_windows_shells() -> list[ShellDescriptor]:
    """Return native Windows shells available on this host."""
    found: list[ShellDescriptor] = []
    for identifier, label in _WINDOWS_CANDIDATES:
        path = shutil.which(identifier)
        if path is None:
            continue
        command: tuple[str, ...]
        command = (path, "-NoLogo") if identifier in {"pwsh", "powershell"} else (path,)
        found.append(
            ShellDescriptor(identifier=identifier, label=label, command=command),
        )
    return found


def _wsl_executable() -> str | None:
    """Return the path to ``wsl`` when present, else ``None``."""
    return shutil.which("wsl")


def _run_text_command(args: list[str]) -> str | None:
    """Run ``args`` and return decoded stdout, or ``None`` on failure.

    Output is decoded as UTF-16-LE first (the Windows ``wsl`` default
    encoding) and falls back to UTF-8. NULs are stripped because some
    Windows console pipelines pad lines with them.
    """
    try:
        completed = subprocess.run(  # noqa: S603 - args are caller-controlled
            args,
            capture_output=True,
            check=False,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _logger.debug("shell discovery command failed: %s (%s)", args, exc)
        return None
    if completed.returncode != 0:
        _logger.debug("shell discovery command non-zero: %s", args)
        return None
    raw = completed.stdout or b""
    for encoding in ("utf-16-le", "utf-8"):
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        return text.replace("\x00", "")
    return None


def _discover_wsl_distros() -> list[ShellDescriptor]:
    """Return WSL distributions reachable via the ``wsl`` launcher."""
    wsl = _wsl_executable()
    if wsl is None:
        return []
    output = _run_text_command([wsl, "--list", "--quiet"])
    if output is None:
        return []

    distros = [line.strip() for line in output.splitlines() if line.strip()]
    found: list[ShellDescriptor] = []
    for distro in distros:
        found.append(
            ShellDescriptor(
                identifier=f"wsl:{distro}",
                label=f"WSL: {distro}",
                command=(wsl, "-d", distro),
            ),
        )
    return found


__all__ = [
    "ShellDescriptor",
    "discover_shells",
]
