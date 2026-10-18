"""Launch failure message formatting helpers.

Refs #13023: surfaces searched paths and submodule remedy in launch failure toasts.
Adheres to Law of Demeter and keeps launcher_model_handlers.py below budget.
"""

from __future__ import annotations

from typing import Any

from src.launchers.task_launch_truthfulness import LaunchDisposition


def format_source_root_not_found(exc: BaseException) -> str:
    """Format a SourceRootNotFoundError or related exception for toast/status messages.

    Adheres to Law of Demeter by accepting the exception directly rather than
    requiring callers to inspect internal exception attributes.
    """
    return str(exc).strip()


def format_truthfulness_audit_message(audit: Any) -> str | None:
    """Format launch disposition explanation for truthfulness audit."""
    if audit is None:
        return None
    disposition = getattr(audit, "disposition", None)
    if disposition in (
        LaunchDisposition.LIBRARY_ONLY,
        LaunchDisposition.PARAMETRIC_CLI,
    ):
        return f"{audit.label}: {audit.explanation} Next action: {audit.next_action}"
    return None


def format_physics_informed_status(name: str, mode: str, package: str) -> str:
    """Return user-facing explanation why physics-informed tile cannot launch."""
    return (
        f"{name} (mode={mode}) has no interactive UI yet. The physics-informed "
        f"models are available as a library only ({package}/); see epic "
        "#5419. Tracking issue for a launcher front-end: #7984."
    )
