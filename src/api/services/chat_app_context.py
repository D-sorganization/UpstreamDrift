"""Shared chat app-context schema and providers (issue #7453).

Defines the single context contract injected into AI chat sessions so the
desktop (PyQt6 Sidekick) and API-server (web chat) providers cannot drift.

The legacy desktop contract from issue #5470 is
:func:`src.shared.python.app_state.agent_context`, which emits
``{"events", "last_diagnostics", "summary"}``. :class:`ChatAppContext` is a
strict superset of those keys, adding live engine/simulation fields filled
from the API server's ``EngineManager`` and ``SimulationService``.

The module deliberately lives under ``src/api/services/`` (not
``src/shared/python/chat/`` or ``ai/``) because those packages are vendored
read-only copies of the Tools repository.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from src.shared.python.app_state import agent_context, get_agent_state_store
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

#: Keys emitted by the legacy desktop provider (``agent_context``, #5470).
#: ``ChatAppContext`` must always be a superset of these.
LEGACY_CONTEXT_KEYS: frozenset[str] = frozenset(
    {"events", "last_diagnostics", "summary"}
)

_DEFAULT_MAX_EVENTS = 50


class SimulationRunContext(BaseModel):
    """Last/current simulation run as seen by the chat assistant."""

    engine: str | None = None
    model: str | None = None
    duration_seconds: float | None = None
    status: str | None = None
    frames: int | None = None
    finished_at: str | None = None
    error: str | None = None
    analysis_summary: str | None = None


class ChatAppContext(BaseModel):
    """Single context contract shared by desktop and API chat providers.

    Postcondition: ``set(model_dump()) >= LEGACY_CONTEXT_KEYS`` so existing
    consumers of the #5470 ``agent_context`` contract keep working.
    """

    engines_loaded: list[str] = Field(default_factory=list)
    active_engine: str | None = None
    active_model: str | None = None
    simulation: SimulationRunContext | None = None
    analysis_summary: str | None = None
    # Legacy agent_context (#5470) keys — preserved verbatim.
    events: list[dict[str, Any]] = Field(default_factory=list)
    last_diagnostics: list[dict[str, Any]] = Field(default_factory=list)
    summary: str = ""


def _engine_fields(engine_manager: Any) -> tuple[list[str], str | None]:
    """Extract (engines_loaded, active_engine) from an EngineManager."""
    if engine_manager is None:
        return [], None
    try:
        info = engine_manager.get_engine_info()
        engines = [str(e) for e in info.get("available_engines", [])]
        current = info.get("current_engine")
        return engines, str(current) if current else None
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        logger.warning("chat_app_context: engine info unavailable: %s", exc)
        return [], None


def _simulation_fields(simulation_service: Any) -> SimulationRunContext | None:
    """Extract the last-run context from a SimulationService, if any."""
    if simulation_service is None:
        return None
    try:
        last_run = simulation_service.stats.last_run
    except (AttributeError, TypeError, RuntimeError) as exc:
        logger.warning("chat_app_context: simulation stats unavailable: %s", exc)
        return None
    if not isinstance(last_run, dict):
        return None
    return SimulationRunContext.model_validate(
        {k: last_run.get(k) for k in SimulationRunContext.model_fields}
    )


def build_chat_app_context(
    engine_manager: Any = None,
    simulation_service: Any = None,
    store: Any = None,
    max_events: int = _DEFAULT_MAX_EVENTS,
) -> ChatAppContext:
    """Build the chat context from live API-server state.

    Every input is optional; missing or failing sources degrade to empty
    fields so the chat session never breaks on context assembly.

    Args:
        engine_manager: ``EngineManager`` (or compatible) exposing
            ``get_engine_info()``.
        simulation_service: ``SimulationService`` (or compatible) exposing
            ``stats.last_run``.
        store: ``HistoryStore`` for recent events; defaults to the
            process-level agent state store.
        max_events: Maximum recent events to include. Must be positive.

    Returns:
        A fully populated :class:`ChatAppContext`.
    """
    if max_events <= 0:
        raise ValueError(f"max_events must be positive, got {max_events}")

    base = agent_context(
        store if store is not None else get_agent_state_store(),
        max_events=max_events,
    )
    engines_loaded, active_engine = _engine_fields(engine_manager)
    simulation = _simulation_fields(simulation_service)

    return ChatAppContext(
        engines_loaded=engines_loaded,
        active_engine=active_engine or (simulation.engine if simulation else None),
        active_model=simulation.model if simulation else None,
        simulation=simulation,
        analysis_summary=simulation.analysis_summary if simulation else None,
        events=base["events"],
        last_diagnostics=base["last_diagnostics"],
        summary=base["summary"],
    )


def make_app_state_provider(
    engine_manager_supplier: Callable[[], Any] | None,
    simulation_service_supplier: Callable[[], Any] | None,
    store_supplier: Callable[[], Any] = get_agent_state_store,
) -> Callable[[], dict[str, Any]]:
    """Build a ``ChatService`` app-state provider from lazy suppliers.

    Suppliers (rather than instances) are accepted so callers can pass
    lazily initialised services (e.g. ``local_server``'s lazy proxies)
    without forcing construction at wiring time.

    Args:
        engine_manager_supplier: Zero-arg callable returning the
            EngineManager, or ``None``.
        simulation_service_supplier: Zero-arg callable returning the
            SimulationService, or ``None``.
        store_supplier: Zero-arg callable returning the HistoryStore.

    Returns:
        Zero-arg callable returning ``ChatAppContext.model_dump()``.

    Raises:
        TypeError: If a non-``None`` supplier is not callable.
    """
    for name, supplier in (
        ("engine_manager_supplier", engine_manager_supplier),
        ("simulation_service_supplier", simulation_service_supplier),
        ("store_supplier", store_supplier),
    ):
        if supplier is not None and not callable(supplier):
            raise TypeError(f"{name} must be callable or None")

    def _provider() -> dict[str, Any]:
        return build_chat_app_context(
            engine_manager=(
                engine_manager_supplier() if engine_manager_supplier else None
            ),
            simulation_service=(
                simulation_service_supplier() if simulation_service_supplier else None
            ),
            store=store_supplier() if store_supplier is not None else None,
        ).model_dump()

    return _provider
