"""Crash-containment policy for the desktop launcher.

Historically the launcher's :func:`sys.excepthook` called
``QApplication.quit()`` unconditionally, so *any* exception that escaped a Qt
slot — including a perfectly recoverable "optional dependency is missing"
failure raised while launching a tile — tore down the whole application the
moment the user dismissed the crash dialog (issues #8066, #8070, #8072,
#8084).

This module holds the pure decision logic so it can be unit-tested without a
running ``QApplication``.  The rule is:

* ``SystemExit`` is the interpreter shutting down on purpose — honour it
  silently.
* Anything raised *before* the launcher window is up cannot be recovered from,
  because there is no usable surface left to return to: report and exit.
* Anything else is contained — report it and keep the launcher alive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

__all__ = [
    "CrashAction",
    "FATAL_EXCEPTION_TYPES",
    "classify_crash",
]


#: Exception types that must always terminate the process regardless of how
#: healthy the UI looks.  ``MemoryError`` is included because continuing after
#: it produces cascading, unactionable failures.
FATAL_EXCEPTION_TYPES: Final[tuple[type[BaseException], ...]] = (
    KeyboardInterrupt,
    MemoryError,
    SystemExit,
)


@dataclass(frozen=True)
class CrashAction:
    """What the exception hook should do about an escaped exception.

    Attributes:
        show_dialog: Whether to present the error to the user.
        quit_application: Whether the application must terminate afterwards.
        title: Dialog title.
        message: Human-readable, non-traceback summary line.
    """

    show_dialog: bool
    quit_application: bool
    title: str
    message: str


_RECOVERABLE = CrashAction(
    show_dialog=True,
    quit_application=False,
    title="Operation Failed",
    message=(
        "UpstreamDrift hit an unexpected error while carrying out that action.\n\n"
        "The launcher is still running — close this dialog and try another tile. "
        "The details below are also written to crash_traceback.txt."
    ),
)

_FATAL_STARTUP = CrashAction(
    show_dialog=True,
    quit_application=True,
    title="Application Crash",
    message=(
        "UpstreamDrift could not finish starting up and must close.\n\n"
        "The details below are also written to crash_traceback.txt."
    ),
)

_SILENT_EXIT = CrashAction(
    show_dialog=False,
    quit_application=True,
    title="",
    message="",
)


def classify_crash(
    exc_type: type[BaseException],
    *,
    launcher_is_alive: bool,
) -> CrashAction:
    """Decide how to handle an exception that escaped a Qt slot.

    Args:
        exc_type: Type of the escaped exception.
        launcher_is_alive: True when a visible top-level launcher window
            exists, i.e. there is still a usable surface to return the user to.

    Returns:
        The :class:`CrashAction` describing the desired handling.

    Raises:
        TypeError: If ``exc_type`` is not an exception class.

    Postcondition:
        ``quit_application`` is always True when ``launcher_is_alive`` is False.
    """
    if not (isinstance(exc_type, type) and issubclass(exc_type, BaseException)):
        raise TypeError(f"exc_type must be an exception class, got {exc_type!r}")

    if issubclass(exc_type, SystemExit):
        return _SILENT_EXIT
    if issubclass(exc_type, FATAL_EXCEPTION_TYPES):
        return _FATAL_STARTUP
    if not launcher_is_alive:
        return _FATAL_STARTUP
    return _RECOVERABLE
