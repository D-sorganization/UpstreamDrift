"""Import-safe diagnostics for the optional PyQt6 chat dock runtime."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class ChatQtDiagnostic:
    """Result of probing whether the Qt chat dock can be imported."""

    available: bool
    reason: str
    detail: str = ""

    def to_dict(self) -> dict[str, str | bool]:
        return {
            "available": self.available,
            "reason": self.reason,
            "detail": self.detail,
        }


def diagnose_chat_qt_runtime(
    *,
    python_executable: str | None = None,
    timeout_seconds: float = 10.0,
) -> ChatQtDiagnostic:
    """Probe PyQt6 in a subprocess and return a structured diagnostic.

    A partially installed PyQt6 runtime can terminate the importing process on
    Windows when Qt DLLs are ABI-incompatible. Probing in a child process keeps
    test collection and host launchers alive while still returning the exact
    import failure text.
    """

    executable = python_executable or sys.executable
    probe = (
        "from PyQt6.QtCore import QCoreApplication; "
        "from PyQt6.QtWebSockets import QWebSocket; "
        "print('ok')"
    )
    try:
        completed = subprocess.run(
            [executable, "-c", probe],
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return ChatQtDiagnostic(
            available=False,
            reason="probe_failed",
            detail=str(exc),
        )
    if completed.returncode == 0:
        return ChatQtDiagnostic(available=True, reason="available")

    detail = "\n".join(
        part.strip()
        for part in (completed.stderr, completed.stdout)
        if part and part.strip()
    )
    return ChatQtDiagnostic(
        available=False,
        reason="import_failed",
        detail=detail or f"probe exited with {completed.returncode}",
    )


__all__ = ["ChatQtDiagnostic", "diagnose_chat_qt_runtime"]
