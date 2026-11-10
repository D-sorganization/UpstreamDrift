"""AIP JSON-RPC method implementations.

Defines all JSON-RPC methods available through the AIP server.
Methods are grouped by namespace: simulation, model, analysis.

See issue #763, honesty slice #14940.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from .dispatcher import MethodRegistry as MethodRegistry
from .errors import AIPMethodNotImplemented

_logger = logging.getLogger(__name__)


def _not_implemented(method: str, detail: str) -> AIPMethodNotImplemented:
    return AIPMethodNotImplemented(
        method,
        f"{method} is not implemented: {detail}",
    )


def create_registry() -> MethodRegistry:
    """Create and populate the AIP method registry.

    Returns:
        Populated MethodRegistry with all AIP methods.
    """
    registry = MethodRegistry()

    # ── Simulation methods ──────────────────────────────────
    registry.register(
        "simulation.start",
        _simulation_start,
        "Start a physics simulation with the given configuration.",
    )
    registry.register(
        "simulation.stop",
        _simulation_stop,
        "Stop the currently running simulation.",
    )
    registry.register(
        "simulation.step",
        _simulation_step,
        "Advance the simulation by one or more timesteps.",
    )
    registry.register(
        "simulation.status",
        _simulation_status,
        "Get current simulation status and metrics.",
    )
    registry.register(
        "simulation.set_control",
        _simulation_set_control,
        "Set actuator control values.",
    )

    # ── Model methods ───────────────────────────────────────
    registry.register(
        "model.load",
        _model_load,
        "Load a URDF/MJCF model file.",
    )
    registry.register(
        "model.query",
        _model_query,
        "Query model properties (joints, links, limits).",
    )
    registry.register(
        "model.list",
        _model_list,
        "List available model files.",
    )

    # ── Analysis methods ────────────────────────────────────
    registry.register(
        "analysis.metrics",
        _analysis_metrics,
        "Get current biomechanics metrics.",
    )
    registry.register(
        "analysis.export",
        _analysis_export,
        "Export analysis data in the specified format.",
    )
    registry.register(
        "analysis.time_series",
        _analysis_time_series,
        "Get time series data for specified metrics.",
    )

    # ── System methods ──────────────────────────────────────
    registry.register(
        "system.capabilities",
        _system_capabilities,
        "Get server capabilities and supported methods.",
    )
    registry.register(
        "system.ping",
        _system_ping,
        "Health check / ping.",
    )

    return registry


_SELF_CHECK_HANDLERS: dict[str, Callable[[Callable[..., Any]], None]] = {
    "system.ping": lambda handler: handler(),
    "model.list": lambda handler: handler(),
}


def methods_passing_self_check(registry: MethodRegistry) -> list[str]:
    """Return method names that are safe to advertise in capabilities (#14940)."""
    supported: list[str] = []
    for name in registry.list_methods():
        if name == "system.capabilities":
            supported.append(name)
        elif (probe := _SELF_CHECK_HANDLERS.get(name)) and (
            h := registry.get_method(name)
        ):
            try:
                probe(h)
                supported.append(name)
            except (AIPMethodNotImplemented, RuntimeError, ValueError, OSError):
                logger.debug("Self-check skipped %s", name)
    return sorted(supported)


# ── Simulation method handlers ──────────────────────────────


def _simulation_start(
    engine_type: str = "mujoco",
    duration: float = 3.0,
    timestep: float = 0.002,
    model_path: str | None = None,
    _context: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Start a simulation.

    Args:
        engine_type: Physics engine to use.
        duration: Simulation duration in seconds.
        timestep: Integration timestep.
        model_path: Optional model file path.
        _context: Injected context with engine_manager.

    Returns:
        Status dict with simulation ID.
    """
    if not (engine_type is not None):
        raise ValueError("engine_type must be provided")
    raise _not_implemented(
        "simulation.start",
        "no SimulationService run was created",
    )


def _simulation_stop(
    _context: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Stop the running simulation.

    Args:
        _context: Injected context.

    Returns:
        Status dict.
    """
    raise _not_implemented(
        "simulation.stop",
        "no SimulationService run exists",
    )


def _simulation_step(
    n_steps: int = 1,
    _context: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Step the simulation forward.

    Args:
        n_steps: Number of timesteps to advance.
        _context: Injected context.

    Returns:
        Step result with state data.
    """
    if not (n_steps is not None):
        raise ValueError("n_steps must be provided")
    raise _not_implemented(
        "simulation.step",
        "no active simulation run to advance",
    )


def _simulation_status(
    _context: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Get simulation status.

    Args:
        _context: Injected context.

    Returns:
        Status information.
    """
    raise _not_implemented(
        "simulation.status",
        "no active simulation run exists",
    )


def _simulation_set_control(
    actuator_index: int = 0,
    value: float = 0.0,
    _context: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Set actuator control value.

    Args:
        actuator_index: Actuator index.
        value: Control value.
        _context: Injected context.

    Returns:
        Acknowledgment.
    """
    if not (actuator_index is not None):
        raise ValueError("actuator_index must be provided")
    raise _not_implemented(
        "simulation.set_control",
        "no active simulation run to control",
    )


# ── Model method handlers ───────────────────────────────────


def _model_load(
    path: str = "",
    _context: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Load a model file.

    Args:
        path: Model file path.
        _context: Injected context.

    Returns:
        Load result with model info.
    """
    if not (path is not None):
        raise ValueError("path must be provided")
    if not path:
        return {"status": "error", "message": "path is required"}
    raise _not_implemented(
        "model.load",
        "no model was loaded into an engine",
    )


def _model_query(
    property_name: str = "joints",
    _context: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Query model properties.

    Args:
        property_name: Property to query (joints, links, limits).
        _context: Injected context.

    Returns:
        Requested property data.
    """
    if not (property_name is not None):
        raise ValueError("property_name must be provided")
    raise _not_implemented(
        "model.query",
        "no loaded model is available to query",
    )


def _model_list(
    _context: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """List available models.

    Args:
        _context: Injected context.

    Returns:
        List of model files.
    """
    from ..routes.models import discover_models

    try:
        models = discover_models()
        return {"models": models, "count": len(models)}
    except (RuntimeError, ValueError, OSError) as exc:
        return {"models": [], "count": 0, "error": str(exc)}


# ── Analysis method handlers ────────────────────────────────


def _analysis_metrics(
    _context: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Get current metrics.

    Args:
        _context: Injected context.

    Returns:
        Current metrics data.
    """
    raise _not_implemented(
        "analysis.metrics",
        "no metrics were read from a simulation recorder",
    )


def _analysis_export(
    format: str = "json",
    _context: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Export analysis data.

    Args:
        format: Export format (json, csv).
        _context: Injected context.

    Returns:
        Export status.
    """
    raise _not_implemented(
        "analysis.export",
        "no export job was enqueued or written",
    )


def _analysis_time_series(
    metrics: list[str] | None = None,
    window: float = 10.0,
    _context: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Get time series data.

    Args:
        metrics: Metric names to include.
        window: Time window in seconds.
        _context: Injected context.

    Returns:
        Time series data.
    """
    raise _not_implemented(
        "analysis.time_series",
        "no time-series recorder is attached to an active run",
    )


# ── System method handlers ──────────────────────────────────


def _system_capabilities(
    _context: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Get server capabilities.

    Args:
        _context: Injected context.

    Returns:
        Capabilities descriptor.
    """
    registry = create_registry()
    supported = methods_passing_self_check(registry)
    ns_map: dict[str, list[str]] = {}
    for n in supported:
        ns_map.setdefault(n.split(".", 1)[0] if "." in n else "root", []).append(n)

    return {
        "server_name": "UpstreamDrift AIP Server",
        "protocol_version": "2.0",
        "capabilities": [
            {"name": k, "version": "1.0", "methods": v}
            for k, v in sorted(ns_map.items())
        ],
        "supported_methods": supported,
    }


def _system_ping(
    _context: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Health check.

    Args:
        _context: Injected context.

    Returns:
        Pong response.
    """
    return {"status": "pong", "server": "UpstreamDrift AIP"}


# ── Helpers ─────────────────────────────────────────────────


def _get_engine_manager(context: dict[str, Any] | None) -> Any:
    """Extract engine_manager from context.

    Args:
        context: Context dict.

    Returns:
        Engine manager or None.
    """
    if context is None:
        return None
    return context.get("engine_manager")
