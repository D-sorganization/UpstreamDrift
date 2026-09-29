# ruff: noqa: E501
"""Workspace / plot slash-command bridge (Tools issue #2849).

Thin ``dock``-taking shims retained for backwards compatibility. The logic now
lives in the headless :class:`chat.workspace_command_handler.WorkspaceCommandHandler`
controller (ADR-0022 / issue #6119); these helpers build a transient controller
bound to the dock's collaborators so any external caller still works and no
logic is duplicated (DRY).
"""

from __future__ import annotations

from typing import Any

from ..workspace_command_handler import (
    WorkspaceCommandHandler,
    build_workspace_context_block,
)

__all__ = [
    "build_workspace_context_block",
    "dispatch_workspace_command",
    "handle_plot",
    "handle_ws_read",
    "handle_ws_write",
]


def _handler_for(dock: Any) -> WorkspaceCommandHandler:
    """Return the dock's controller, or a transient one bound to its collaborators."""
    existing = getattr(dock, "_workspace_commands", None)
    if isinstance(existing, WorkspaceCommandHandler):
        return existing
    return WorkspaceCommandHandler(
        emit=lambda text: dock._add_bubble("assistant", text),
        provider=getattr(dock, "_workspace_provider", None),
        plot_sink=getattr(dock, "_plot_request_sink", None),
    )


def handle_ws_read(dock: Any, arg: str) -> None:
    """``/ws.read NAME`` — read a workspace variable and show its preview."""
    _handler_for(dock).handle_ws_read(arg)


def handle_ws_write(dock: Any, arg: str) -> None:
    """``/ws.write NAME JSON_VALUE`` — write a value into the workspace."""
    _handler_for(dock).handle_ws_write(arg)


def handle_plot(dock: Any, arg: str) -> None:
    """``/plot {json}`` — forward a plot spec to the host sink."""
    _handler_for(dock).handle_plot(arg)


def dispatch_workspace_command(dock: Any, cmd: str, arg: str) -> None:
    """Route ``/ws.read``, ``/ws.write`` and ``/plot`` slash commands."""
    _handler_for(dock).dispatch(cmd, arg)
