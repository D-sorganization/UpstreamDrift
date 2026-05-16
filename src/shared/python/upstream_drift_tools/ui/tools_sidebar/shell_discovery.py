"""Shell discovery utilities for the Sidekick OS terminal tab.

Provides :func:`discover_shells` which enumerates available OS shells
on the current platform by probing :func:`shutil.which`.

Issue #5617: real OS terminal tab with PTY backend and shell switcher.
"""

from __future__ import annotations

import logging
import shutil
import sys
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Platform-specific shell candidates
# ---------------------------------------------------------------------------

_POSIX_CANDIDATES: list[tuple[str, str, list[str]]] = [
    ("bash", "Bash", []),
    ("zsh", "zsh", []),
    ("fish", "fish", []),
    ("sh", "sh", []),
    ("ksh", "ksh", []),
    ("dash", "dash", []),
]

_WINDOWS_CANDIDATES: list[tuple[str, str, list[str]]] = [
    ("pwsh", "PowerShell (pwsh)", ["-NoLogo"]),
    ("powershell", "PowerShell", ["-NoLogo"]),
    ("cmd", "Command Prompt", ["/K"]),
    ("wsl", "WSL (default)", ["-e", "bash"]),
]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ShellDescriptor:
    """Describes a discoverable OS shell.

    Attributes:
        display_name: Human-readable name shown in the shell selector combo box.
        binary: Absolute path to the shell executable (as resolved by PATH).
        args: Additional command-line arguments prepended after the binary.
    """

    display_name: str
    binary: str
    args: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def discover_shells() -> list[ShellDescriptor]:
    """Return available OS shells on the current platform.

    Probes :func:`shutil.which` for each platform-specific candidate.
    Deduplicates by resolved binary path so aliased shells appear only once.

    Returns:
        Ordered list of :class:`ShellDescriptor` instances.  Empty list when
        no shell is found (e.g. restricted environments).

    Postcondition:
        No two entries in the returned list share the same ``binary`` path.
    """
    candidates = _WINDOWS_CANDIDATES if sys.platform == "win32" else _POSIX_CANDIDATES

    seen_binaries: set[str] = set()
    result: list[ShellDescriptor] = []

    for name, display_name, args in candidates:
        path = shutil.which(name)
        if path is None:
            continue
        if path in seen_binaries:
            logger.debug(
                "Skipping duplicate shell binary %r (already registered)", path
            )
            continue
        seen_binaries.add(path)
        result.append(
            ShellDescriptor(display_name=display_name, binary=path, args=args)
        )
        logger.debug("Discovered shell: %s -> %s", display_name, path)

    return result


__all__ = ["ShellDescriptor", "discover_shells"]
