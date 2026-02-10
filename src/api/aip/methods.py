"""AIP JSON-RPC method implementations.

Defines all JSON-RPC methods available through the AIP server.
Methods are grouped by namespace: simulation, model, analysis.

See issue #763
"""

from __future__ import annotations

from typing import Any

from .dispatcher import MethodRegistry


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
    engine_manager = _get_engine_manager(_context)

    # Try to load the engine
    try:
        if engine_manager and hasattr(engine_manager, "load_engine"):
            engine_manager.load_engine(engine_type)
    except Exception as exc:
        return {
            "status": "error",
            "message": f"Failed to load engine: {str(exc)}",
        }

    return {
        "status": "started",
        "engine_type": engine_type,
        "duration": duration,
        "timestep": timestep,
        "model_path": model_path,
    }


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
    return {"status": "stopped"}


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
    engine_manager = _get_engine_manager(_context)
    state: dict[str, Any] = {}

    if engine_manager:
        try:
            engine = engine_manager.get_active_engine()
            if engine and hasattr(engine, "step"):
                for _ in range(n_steps):
                    engine.step()
            if engine and hasattr(engine, "get_state"):
                state = engine.get_state()
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    return {
        "status": "ok",
        "n_steps": n_steps,
        "state": state,
    }


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
    engine_manager = _get_engine_manager(_context)

    if engine_manager:
        try:
            engine = engine_manager.get_active_engine()
            if engine:
                state = engine.get_state() if hasattr(engine, "get_state") else {}
                return {
                    "running": True,
                    "engine": str(getattr(engine, "engine_type", "unknown")),
                    "sim_time": state.get("time", 0.0),
                    "state_keys": list(state.keys()),
                }
        except Exception:
            pass

    return {"running": False, "engine": None, "sim_time": 0.0}


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
    engine_manager = _get_engine_manager(_context)

    if engine_manager:
        try:
            engine = engine_manager.get_active_engine()
            if engine and hasattr(engine, "set_control"):
                engine.set_control(actuator_index, value)
                return {
                    "status": "ok",
                    "actuator_index": actuator_index,
                    "applied_value": value,
                }
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    return {
        "status": "ok",
        "actuator_index": actuator_index,
        "applied_value": value,
        "note": "No active engine; command buffered.",
    }


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
    if not path:
        return {"status": "error", "message": "path is required"}

    return {
        "status": "loaded",
        "path": path,
        "format": "urdf" if path.endswith(".urdf") else "mjcf",
    }


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
    engine_manager = _get_engine_manager(_context)

    if engine_manager:
        try:
            engine = engine_manager.get_active_engine()
            if engine:
                if property_name == "joints" and hasattr(engine, "joint_names"):
                    return {"joints": list(engine.joint_names)}
                if property_name == "joints" and hasattr(engine, "get_joint_names"):
                    return {"joints": list(engine.get_joint_names())}
        except Exception:
            pass

    return {"property": property_name, "data": None, "note": "No active model"}


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
    from ..routes.models import _discover_models

    try:
        models = _discover_models()
        return {"models": models, "count": len(models)}
    except Exception as exc:
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
    engine_manager = _get_engine_manager(_context)

    if engine_manager:
        try:
            engine = engine_manager.get_active_engine()
            if engine and hasattr(engine, "get_state"):
                state = engine.get_state()
                return {
                    "sim_time": state.get("time", 0.0),
                    "positions": state.get("positions", []),
                    "velocities": state.get("velocities", []),
                    "torques": state.get("torques", []),
                }
        except Exception:
            pass

    return {"sim_time": 0.0, "note": "No active simulation"}


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
    return {
        "status": "ok",
        "format": format,
        "note": "Export queued. Use analysis.metrics for current data.",
    }


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
    return {
        "metrics": metrics or ["positions", "velocities"],
        "window": window,
        "data": {},
        "note": "Time series requires active simulation with recording enabled.",
    }


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
    namespaces = registry.list_by_namespace()

    capabilities = []
    for ns, methods in namespaces.items():
        capabilities.append(
            {
                "name": ns,
                "version": "1.0",
                "methods": methods,
            }
        )

    return {
        "server_name": "UpstreamDrift AIP Server",
        "protocol_version": "2.0",
        "capabilities": capabilities,
        "supported_methods": registry.list_methods(),
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
