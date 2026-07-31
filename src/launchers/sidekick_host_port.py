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

from shared.python.sidekick.agent.subtab_adapter import (
    CalculatorRun,
    StateProfile,
    SubtabAdapter,
    WorkspaceSnapshot,
)
from shared.python.sidekick.agent.action_service import SidekickActionService
from shared.python.sidekick.agent.host_adapter import (
    HostAdapter,
    HostCapability,
    HostInvocationResult,
)

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

__all__ = [
    "LauncherHostActionPort",
    "LauncherSubtabPort",
    "TabHost",
    "create_launcher_action_service",
    "create_launcher_host_adapter",
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
        calculators: (
            Mapping[str, Callable[[Mapping[str, Any]], CalculatorRun]] | None
        ) = None,
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


class LauncherHostActionPort:
    """``HostActionPort`` for launcher-wide actions.

    This port deliberately exposes only the launcher's public command
    surface. It does not inspect widgets or tab internals; tab-scoped work
    stays in :class:`LauncherSubtabPort`.
    """

    host_id = "launcher"

    def __init__(self, launcher: Any) -> None:
        if launcher is None:
            raise ValueError("launcher must be provided")
        self._launcher = launcher

    def list_capabilities(self) -> Sequence[HostCapability]:
        return (
            HostCapability(
                capability_id="host.launcher.list_tiles",
                summary="List launcher tile ids available to open.",
                params_schema={"type": "object", "properties": {}, "required": []},
            ),
            HostCapability(
                capability_id="host.launcher.open_tile",
                summary="Open or focus a launcher tile by id.",
                params_schema={
                    "type": "object",
                    "properties": {
                        "tool_id": {
                            "type": "string",
                            "description": "Launcher tile or Sidekick tab id to open.",
                        }
                    },
                    "required": ["tool_id"],
                },
            ),
        )

    def invoke(
        self, capability_id: str, params: Mapping[str, Any]
    ) -> HostInvocationResult:
        if capability_id == "host.launcher.list_tiles":
            return HostInvocationResult(ok=True, value=self._list_tiles())
        if capability_id == "host.launcher.open_tile":
            tool_id = params.get("tool_id")
            if not isinstance(tool_id, str) or not tool_id.strip():
                return HostInvocationResult(
                    ok=False, error="tool_id must be a non-empty string"
                )
            return self._open_tile(tool_id.strip())
        return HostInvocationResult(
            ok=False, error=f"unknown launcher capability: {capability_id!r}"
        )

    def _list_tiles(self) -> list[str]:
        orchestrator = getattr(self._launcher, "orchestrator", None)
        available = getattr(orchestrator, "available_models", None)
        if isinstance(available, Mapping):
            return sorted(str(key) for key in available)
        if isinstance(available, Sequence) and not isinstance(available, str):
            out: list[str] = []
            for item in available:
                tile_id = getattr(item, "id", item)
                if isinstance(tile_id, str):
                    out.append(tile_id)
            return sorted(out)

        host = getattr(self._launcher, "embedded_host", None)
        open_tool_ids = getattr(host, "open_tool_ids", None)
        if callable(open_tool_ids):
            return sorted(str(tool_id) for tool_id in open_tool_ids())
        return []

    def _open_tile(self, tool_id: str) -> HostInvocationResult:
        opener = getattr(self._launcher, "open_sidekick_tab", None)
        if not callable(opener):
            return HostInvocationResult(
                ok=False, error="launcher does not expose open_sidekick_tab"
            )
        try:
            opener(tool_id)
        except (RuntimeError, ValueError, KeyError) as exc:
            return HostInvocationResult(ok=False, error=str(exc))
        return HostInvocationResult(ok=True, value={"opened": tool_id})


def create_launcher_subtab_adapter(
    host: TabHost,
    *,
    workspace: Any | None = None,
    calculators: (
        Mapping[str, Callable[[Mapping[str, Any]], CalculatorRun]] | None
    ) = None,
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


def create_launcher_host_adapter(launcher: Any) -> HostAdapter:
    """Build the ready-to-register ``HostAdapter`` for launcher actions."""
    return HostAdapter(port=LauncherHostActionPort(launcher))


def create_launcher_action_service(
    *,
    launcher: Any,
    embedded_host: TabHost | None = None,
    workspace: Any | None = None,
    calculators: (
        Mapping[str, Callable[[Mapping[str, Any]], CalculatorRun]] | None
    ) = None,
    profile_path: str | Path | None = None,
) -> SidekickActionService:
    """Create the launcher window's Sidekick action service.

    The subtab namespace is registered only when an embedded host is
    available. Host actions remain available either way so older launcher
    layouts that do not yet construct ``EmbeddedHostWidget`` keep working.
    """
    service = SidekickActionService()
    if embedded_host is not None:
        service.register(
            create_launcher_subtab_adapter(
                embedded_host,
                workspace=workspace,
                calculators=calculators,
                profile_path=profile_path,
            )
        )
    service.register(create_launcher_host_adapter(launcher))
    return service
