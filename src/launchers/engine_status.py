"""Engine status-label text and reset helpers (issue #12962, refs #12678).

The launcher status label reads ``* MuJoCo Running`` while an engine tab is
open. Closing the tab used to leave that text behind. This module owns the
text format (one place, DRY) and the "reset when that engine's tab closes"
rule so the launcher only wires a signal to it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from src.launchers.managed_process import ManagedProcess, ProcessKind, ProcessState

__all__ = [
    "apply_managed_process_transition",
    "apply_post_launch_for_manager",
    "apply_post_launch_status",
    "engine_crashed_text",
    "engine_running_text",
    "engine_starting_text",
    "is_engine_running_status",
    "make_qt_process_changed_handler",
    "reset_status_if_engine_closed",
]

_RUNNING_PREFIX_CHARS = "*> \t"


class _TextLabel(Protocol):
    def text(self) -> str: ...


def _format_engine_state_text(name: str, verb: str, suffix: str = "") -> str:
    if not isinstance(name, str):
        raise TypeError("name must be a string")
    if not name.strip():
        raise ValueError("name must not be blank")
    if not isinstance(suffix, str):
        raise TypeError("suffix must be a string")
    tail = f" ({suffix})" if suffix else ""
    return f"* {name} {verb}{tail}"


def engine_running_text(name: str, suffix: str = "") -> str:
    """Return the status-label text for a running engine (``* Name Running``)."""
    return _format_engine_state_text(name, "Running", suffix)


def is_engine_running_status(text: str, name: str) -> bool:
    """Return True when ``text`` is any ``Running`` status for engine ``name``."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(name, str):
        raise TypeError("name must be a string")
    if not name.strip():
        return False
    body = text.lstrip(_RUNNING_PREFIX_CHARS)
    if not body.startswith(name):
        return False
    rest = body[len(name) :]
    return rest == " Running" or rest.startswith(" Running ")


def reset_status_if_engine_closed(
    label: _TextLabel,
    closed_title: str,
    restore_idle: Callable[[], None],
) -> bool:
    """Call ``restore_idle`` if ``label`` still says ``closed_title`` is running.

    Returns True when the status was reset. A label showing a different
    engine's status is left untouched.
    """
    if label is None:
        raise ValueError("label must be provided")
    if not callable(restore_idle):
        raise TypeError("restore_idle must be callable")
    if not is_engine_running_status(label.text(), closed_title):
        return False
    restore_idle()
    return True


def engine_starting_text(name: str, suffix: str = "") -> str:
    """Return the status-label text while a child launch is starting."""
    return _format_engine_state_text(name, "Starting", suffix)


def engine_crashed_text(name: str, exit_code: int | None) -> str:
    """Return the status-label text for a crashed child launch."""
    code = exit_code if exit_code is not None else 1
    return f"* {name} Crashed (exit {code})"


def _format_stderr_detail(tail: str) -> str:
    lines = [line.strip() for line in tail.splitlines() if line.strip()]
    if not lines:
        return ""
    return f": {lines[-1]}"


def _kind_suffix(kind: ProcessKind) -> str:
    if kind == ProcessKind.WSL:
        return "WSL"
    if kind == ProcessKind.DOCKER:
        return "Docker"
    return ""


def apply_post_launch_status(
    managed: Any,
    name: str,
    suffix: str,
    set_status: Callable[[str, str], None],
    show_toast: Callable[[str, str], None],
    *,
    success_style: str,
    error_style: str,
    toast_label: str,
) -> None:
    """Set honest post-launch status and toast from ``ManagedProcess`` state."""
    if isinstance(managed, ManagedProcess):
        if managed.state == ProcessState.CRASHED:
            detail = _format_stderr_detail(managed.last_stderr_tail)
            show_toast(f"{name} exited ({managed.exit_code}){detail}", "error")
            set_status(engine_crashed_text(name, managed.exit_code), error_style)
            return
        if managed.state == ProcessState.STARTING:
            show_toast(toast_label, "success")
            set_status(engine_starting_text(name, suffix), success_style)
            return
    show_toast(toast_label, "success")
    set_status(engine_running_text(name, suffix), success_style)


def apply_managed_process_transition(
    managed: ManagedProcess,
    set_status: Callable[[str, str], None],
    show_toast: Callable[[str, str], None],
    restore_idle: Callable[[], None],
    *,
    success_style: str,
    error_style: str,
) -> None:
    """Reflect a ``ManagedProcess`` state transition into the launcher status bar."""
    suffix = _kind_suffix(managed.kind)
    if managed.state == ProcessState.STARTING:
        set_status(engine_starting_text(managed.key, suffix), success_style)
    elif managed.state == ProcessState.RUNNING:
        set_status(engine_running_text(managed.key, suffix), success_style)
    elif managed.state == ProcessState.CRASHED:
        detail = _format_stderr_detail(managed.last_stderr_tail)
        show_toast(f"{managed.key} exited ({managed.exit_code}){detail}", "error")
        set_status(engine_crashed_text(managed.key, managed.exit_code), error_style)
    elif managed.state == ProcessState.EXITED:
        restore_idle()


def apply_post_launch_for_manager(
    process_manager: Any,
    name: str,
    suffix: str,
    set_status: Callable[[str, str], None],
    show_toast: Callable[[str, str], None],
    *,
    success_style: str,
    error_style: str,
    toast_label: str,
) -> None:
    """Look up ``name`` on ``process_manager`` and apply post-launch status."""
    getter = getattr(process_manager, "get_managed_process", None)
    managed = getter(name) if callable(getter) else None
    apply_post_launch_status(
        managed,
        name,
        suffix,
        set_status,
        show_toast,
        success_style=success_style,
        error_style=error_style,
        toast_label=toast_label,
    )


def make_qt_process_changed_handler(
    schedule_fn: Callable[[int, Callable[[], None]], object],
    set_status: Callable[[str, str], None],
    show_toast: Callable[[str, str], None],
    restore_idle: Callable[[], None],
    *,
    success_style: str,
    error_style: str,
) -> Callable[[ManagedProcess], None]:
    """Return a ProcessManager callback that dispatches transitions onto the UI thread."""

    def _on_changed(managed: ManagedProcess) -> None:
        def _apply() -> None:
            apply_managed_process_transition(
                managed,
                set_status,
                show_toast,
                restore_idle,
                success_style=success_style,
                error_style=error_style,
            )

        schedule_fn(0, _apply)

    return _on_changed
