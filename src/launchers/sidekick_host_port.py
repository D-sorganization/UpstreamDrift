"""Launcher-side ``SubtabActionPort`` implementation (epic #5967 follow-up).

The Sidekick agent layer (``sidekick.agent``) ships a fully tested
``SubtabAdapter`` whose only dependency is the narrow
:class:`~sidekick.agent.subtab_adapter.SubtabActionPort` Protocol. Until
now no host implemented that port, so chat agents could not see or
control launcher tabs. This module closes that gap for the PyQt6
launcher: :class:`LauncherSubtabPort` bridges the port onto
:class:`~src.launchers.embedded_host.EmbeddedHostWidget` plus an
injected workspace registry, calculator table, and state-profile store.

Design notes:

* **No PyQt6 imports.** The port only calls duck-typed host methods
  (``open_tool_ids``, ``focus_tab``, ``open_tab``, ``close_tab``,
  ``backgrounded_tools``, ``popped_out_tools``), so it can be unit
  tested headlessly against a fake host.
* **LOD.** The port never reaches into host internals; everything goes
  through the host's public tab API.
* **DbC.** Construction validates the host exposes the required
  surface; tab methods raise ``KeyError`` for unknown tabs as the
  Protocol requires.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from sidekick.agent.subtab_adapter import (
    CalculatorRun,
    StateProfile,
    SubtabAdapter,
    WorkspaceSnapshot,
)

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

__all__ = [
    "LauncherSubtabPort",
    "TabHost",
    "create_launcher_subtab_adapter",
]


@runtime_checkable
class TabHost(Protocol):
    """The slice of ``EmbeddedHostWidget`` the port needs.

    Kept as a Protocol so tests can pass a fake and so the port stays
    importable in headless environments where PyQt6 is unavailable.
    """

    def open_tool_ids(self) -> list[str]: ...

    def active_tool_id(self) -> str | None: ...

    def focus_tab(self, tool_id: str) -> None: ...

    def open_tab(self, tool_id: str) -> int: ...

    def close_tab(self, target: int | str, *, destroy: bool = True) -> bool: ...

    def backgrounded_tools(self) -> set[str]: ...

    def popped_out_tools(self) -> set[str]: ...


class _DictWorkspace:
    """Minimal in-memory workspace used when no registry is injected."""

    def __init__(self) -> None:
        self._values: dict[str, Any] = {}

    def list(self) -> list[str]:
        return sorted(self._values)

    def get(self, name: str, default: Any = None) -> Any:
        return self._values.get(name, default)

    def set(self, name: str, value: Any) -> None:
        self._values[name] = value


class LauncherSubtabPort:
    """``SubtabActionPort`` backed by the launcher's embedded tool host.

    Args:
        host: The tab host (normally an ``EmbeddedHostWidget``).
        workspace: Optional workspace registry exposing ``list()``,
            ``get(name, default)``, and ``set(name, value)`` (the
            ``sidekick.ui.tools_sidebar.registry.WorkspaceRegistry``
            surface). Defaults to an in-memory store.
        calculators: Optional mapping of calculator id to a callable
            ``(inputs) -> CalculatorRun``.
        profile_path: Optional JSON file used to persist state profiles
            across sessions. Profiles stay in memory when omitted.

    Raises:
        TypeError: If ``host`` does not satisfy :class:`TabHost`.
    """

    def __init__(
        self,
        host: TabHost,
        *,
        workspace: Any | None = None,
        calculators: Mapping[str, Callable[[Mapping[str, Any]], CalculatorRun]]
        | None = None,
        profile_path: str | Path | None = None,
    ) -> None:
        if not isinstance(host, TabHost):
            raise TypeError(f"host must satisfy TabHost, got {type(host).__name__}")
        self._host = host
        self._workspace = workspace if workspace is not None else _DictWorkspace()
        self._calculators = dict(calculators or {})
        self._profile_path = Path(profile_path) if profile_path else None
        self._profiles: dict[str, dict[str, Any]] = self._load_profiles()

    # ---- Tabs ------------------------------------------------------------

    def list_tabs(self) -> Sequence[str]:
        """Open tabs in display order, then backgrounded and popped-out."""
        open_ids = self._host.open_tool_ids()
        seen = set(open_ids)
        extras = sorted(
            (self._host.backgrounded_tools() | self._host.popped_out_tools()) - seen
        )
        return [*open_ids, *extras]

    def active_tab(self) -> str | None:
        return self._host.active_tool_id()

    def focus(self, tab_id: str) -> None:
        self._host.focus_tab(tab_id)

    def set_visible(self, tab_id: str, visible: bool) -> None:
        if visible:
            if (
                tab_id
                in self._host.backgrounded_tools()
                | set(self._host.open_tool_ids())
                | self._host.popped_out_tools()
            ):
                self._host.open_tab(tab_id)
                return
            try:
                self._host.open_tab(tab_id)
            except ValueError as exc:
                # The Protocol promises KeyError for unknown tabs.
                raise KeyError(tab_id) from exc
            return
        if tab_id in self._host.backgrounded_tools():
            return  # already hidden — idempotent
        if not self._host.close_tab(tab_id, destroy=False):
            raise KeyError(tab_id)

    # ---- Workspace --------------------------------------------------------

    def workspace_snapshot(self) -> WorkspaceSnapshot:
        values = {name: self._workspace.get(name) for name in self._workspace.list()}
        return WorkspaceSnapshot(values=values)

    def workspace_set_variable(self, name: str, value: Any) -> Any:
        prior = self._workspace.get(name)
        self._workspace.set(name, value)
        return prior

    # ---- Calculators -------------------------------------------------------

    def register_calculator(
        self, calculator_id: str, fn: Callable[[Mapping[str, Any]], CalculatorRun]
    ) -> None:
        """Expose a named calculator to chat agents.

        Raises:
            ValueError: If ``calculator_id`` is empty or already registered.
        """
        if not calculator_id or not calculator_id.strip():
            raise ValueError("calculator_id must be a non-empty string")
        if calculator_id in self._calculators:
            raise ValueError(f"calculator {calculator_id!r} is already registered")
        self._calculators[calculator_id] = fn

    def calculator_run(
        self, calculator_id: str, inputs: Mapping[str, Any]
    ) -> CalculatorRun:
        fn = self._calculators.get(calculator_id)
        if fn is None:
            raise KeyError(calculator_id)
        return fn(inputs)

    # ---- State profiles -----------------------------------------------------

    def state_profile_save(self, name: str, payload: Mapping[str, Any]) -> None:
        self._profiles[name] = dict(payload)
        self._persist_profiles()

    def state_profile_load(self, name: str) -> StateProfile:
        if name not in self._profiles:
            raise KeyError(name)
        return StateProfile(name=name, payload=dict(self._profiles[name]))

    def state_profile_delete(self, name: str) -> None:
        self._profiles.pop(name, None)
        self._persist_profiles()

    # ---- Persistence (internal) ---------------------------------------------

    def _load_profiles(self) -> dict[str, dict[str, Any]]:
        if self._profile_path is None or not self._profile_path.exists():
            return {}
        try:
            raw = json.loads(self._profile_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.exception(
                "Failed to load sidekick state profiles from %s", self._profile_path
            )
            return {}
        if not isinstance(raw, dict):
            logger.warning(
                "Ignoring malformed profile store %s (not an object)",
                self._profile_path,
            )
            return {}
        return {
            str(name): dict(payload)
            for name, payload in raw.items()
            if isinstance(payload, dict)
        }

    def _persist_profiles(self) -> None:
        if self._profile_path is None:
            return
        try:
            self._profile_path.parent.mkdir(parents=True, exist_ok=True)
            self._profile_path.write_text(
                json.dumps(self._profiles, indent=2, default=repr),
                encoding="utf-8",
            )
        except OSError:
            logger.exception(
                "Failed to persist sidekick state profiles to %s", self._profile_path
            )


def create_launcher_subtab_adapter(
    host: TabHost,
    *,
    workspace: Any | None = None,
    calculators: Mapping[str, Callable[[Mapping[str, Any]], CalculatorRun]]
    | None = None,
    profile_path: str | Path | None = None,
) -> SubtabAdapter:
    """Build the ready-to-register ``SidekickActionHandler`` for a host.

    Convenience factory so launcher wiring is one line::

        service.register(create_launcher_subtab_adapter(embedded_host))

    Returns:
        A :class:`SubtabAdapter` whose port is a
        :class:`LauncherSubtabPort` over ``host``.
    """
    port = LauncherSubtabPort(
        host,
        workspace=workspace,
        calculators=calculators,
        profile_path=profile_path,
    )
    return SubtabAdapter(port=port)
