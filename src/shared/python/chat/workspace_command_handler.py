# ruff: noqa: E501
"""Headless workspace / plot slash-command controller (ADR-0022, issue #6119).

Extracted from ``ChatDockWidget`` as a self-contained, Qt-free controller so
the workspace bridge logic can be unit-tested without a ``QApplication`` (this
sidesteps the Sidekick multi-widget Qt segfault). The widget owns one instance
and delegates ``/ws.read``, ``/ws.write`` and ``/plot`` slash commands to it
(composition, not inheritance).

The controller takes flat, typed collaborators only — a workspace provider, a
plot-request sink, and an ``emit`` callback that renders an assistant bubble —
so it never reaches back into the view (LOD). Behaviour is byte-for-byte
identical to the previous free helpers in ``chat._qt.workspace``; those helpers
now delegate here so the logic lives in exactly one place (DRY).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from ._workspace_protocol import WorkspaceVariableInfo

logger = logging.getLogger(__name__)

__all__ = ["WorkspaceCommandHandler", "build_workspace_context_block"]

# Preview-length cap for ``/ws.read`` output. Kept as a module constant so the
# truncation contract is asserted in tests without duplicating the literal.
_READ_PREVIEW_MAX = 200


def build_workspace_context_block(provider: Any) -> str:
    """Return a bounded system-prompt fragment listing workspace vars.

    Returns an empty string when no provider is wired so the dock's outbound
    payload stays byte-for-byte identical to the pre-bridge shape. Provider
    failures are swallowed (the host adapter must never crash chat).
    """
    if provider is None:
        return ""
    try:
        variables = provider.describe()
    except Exception:  # noqa: BLE001 - host adapter must not crash chat
        logger.exception("workspace provider describe() failed")
        return ""
    if not variables:
        return ""

    lines = ["Available workspace variables:"]
    for info in variables:
        if not isinstance(info, WorkspaceVariableInfo):
            continue
        shape_str = (
            ", ".join(str(dim) for dim in info.shape)
            if info.shape is not None
            else "scalar"
        )
        lines.append(
            f"- {info.name}: {info.dtype}, shape ({shape_str}), "
            f'preview="{info.preview}"'
        )
    return "\n".join(lines)


class WorkspaceCommandHandler:
    """Route ``/ws.read``, ``/ws.write`` and ``/plot`` slash commands.

    Collaborators are supplied as plain callables / objects so the handler is
    fully headless:

    Args:
        emit: Callback that renders an assistant message (the view supplies
            ``lambda text: self._add_bubble("assistant", text)``). Must be
            callable.
        provider: Optional workspace provider exposing ``describe``/``read``/
            ``write``. ``None`` means the bridge is unavailable.
        plot_sink: Optional callable accepting a parsed plot spec. ``None``
            means the plot tab is unavailable.
    """

    def __init__(
        self,
        *,
        emit: Callable[[str], None],
        provider: Any | None = None,
        plot_sink: Callable[[Any], None] | None = None,
    ) -> None:
        if not callable(emit):
            raise TypeError("WorkspaceCommandHandler: emit must be callable")
        self._emit = emit
        self.provider = provider
        self.plot_sink = plot_sink

    def context_block(self) -> str:
        """Return the bounded workspace-variable system-prompt fragment."""
        return build_workspace_context_block(self.provider)

    def dispatch(self, cmd: str, arg: str) -> None:
        """Route a parsed slash command to its handler.

        DbC:
            Pre: ``cmd`` is one of ``"ws.read"``, ``"ws.write"``, ``"plot"``.
        """
        if cmd == "ws.read":
            self.handle_ws_read(arg)
            return
        if cmd == "ws.write":
            self.handle_ws_write(arg)
            return
        if cmd == "plot":
            self.handle_plot(arg)
            return
        raise ValueError(f"unknown workspace command: {cmd}")

    def handle_ws_read(self, arg: str) -> None:
        """``/ws.read NAME`` — read a workspace variable and show its preview."""
        name = arg.strip()
        if not name:
            self._emit("Usage: /ws.read NAME")
            return
        if self.provider is None:
            self._emit("Workspace bridge not available in this chat.")
            return
        try:
            value = self.provider.read(name)
        except KeyError:
            self._emit(f"Workspace variable not found: {name}")
            return
        except Exception as exc:  # noqa: BLE001 - host adapter errors
            logger.exception("workspace read failed for %s", name)
            self._emit(f"Workspace read failed: {exc}")
            return
        preview = repr(value)
        if len(preview) > _READ_PREVIEW_MAX:
            preview = preview[: _READ_PREVIEW_MAX - 3] + "..."
        self._emit(f"{name} = {preview}")

    def handle_ws_write(self, arg: str) -> None:
        """``/ws.write NAME JSON_VALUE`` — write a value into the workspace."""
        parts = arg.split(maxsplit=1)
        if len(parts) != 2:
            self._emit("Usage: /ws.write NAME JSON_VALUE")
            return
        name, raw_value = parts[0].strip(), parts[1].strip()
        if not name:
            self._emit("Usage: /ws.write NAME JSON_VALUE")
            return
        if self.provider is None:
            self._emit("Workspace bridge not available in this chat.")
            return
        try:
            value = json.loads(raw_value)
        except (json.JSONDecodeError, TypeError) as exc:
            self._emit(f"Could not parse JSON value: {exc}")
            return
        try:
            self.provider.write(name, value)
        except TypeError as exc:
            self._emit(f"Workspace write rejected: {exc}")
            return
        except Exception as exc:  # noqa: BLE001 - host adapter errors
            logger.exception("workspace write failed for %s", name)
            self._emit(f"Workspace write failed: {exc}")
            return
        self._emit(f"Wrote workspace variable: {name}")

    def handle_plot(self, arg: str) -> None:
        """``/plot {json}`` — forward a plot spec to the host sink."""
        spec_text = arg.strip()
        if not spec_text:
            self._emit("Usage: /plot {json plot spec}")
            return
        if self.plot_sink is None:
            self._emit("Plot tab not available in this chat.")
            return
        try:
            spec = json.loads(spec_text)
        except (json.JSONDecodeError, TypeError) as exc:
            self._emit(f"Could not parse plot spec JSON: {exc}")
            return
        try:
            self.plot_sink(spec)
        except Exception as exc:  # noqa: BLE001 - host adapter errors
            logger.exception("plot request sink failed")
            self._emit(f"Plot request failed: {exc}")
            return
        self._emit("Plot request submitted.")
