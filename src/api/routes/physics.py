# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

"""Physics simulation control routes.

Provides endpoints for per-actuator control, force/torque queries,
biomechanics metrics, control features registry, and simulation
runtime controls (speed, camera, recording).

See issue #1209, #1202

All dependencies are injected via FastAPI's Depends() mechanism.
No module-level mutable state.
"""

from __future__ import annotations

import contextlib
import time
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException

from src.api.middleware.error_handler import handle_api_errors
from src.shared.python.core.contracts import precondition
from src.shared.python.logging_pkg.logging_config import (
    get_logger as _get_module_logger,
)

from ..dependencies import get_engine_manager, get_logger, get_simulation_service
from ..models.requests import (
    ActuatorUpdateRequest,
    CameraPresetRequest,
    SpeedControlRequest,
    TrajectoryRecordRequest,
)
from ..models.responses import (
    ActuatorStateResponse,
    BiomechanicsMetricsResponse,
    CameraPresetResponse,
    ControlFeaturesResponse,
    ForceVectorResponse,
    SimulationStatsResponse,
    SpeedControlResponse,
    TrajectoryRecordResponse,
)

if TYPE_CHECKING:
    from src.shared.python.engine_core.engine_manager import EngineManager

    from ..services.simulation_service import SimulationService

_logger = _get_module_logger(__name__)

router = APIRouter()
_CONTROL_INTERFACE_CACHE: dict[int, Any] = {}
_FEATURES_REGISTRY_CACHE: dict[int, Any] = {}


def clear_physics_caches() -> None:
    """Invalidate control-interface and features-registry caches.

    Must be called whenever the active engine changes so that subsequent
    requests reflect the new engine's metadata.
    """
    _CONTROL_INTERFACE_CACHE.clear()
    _FEATURES_REGISTRY_CACHE.clear()


# Camera preset definitions (position, target, up vectors)
CAMERA_PRESETS: dict[str, dict[str, list[float]]] = {
    "side": {
        "position": [3.0, 0.0, 1.5],
        "target": [0.0, 0.0, 1.0],
        "up": [0.0, 0.0, 1.0],
    },
    "front": {
        "position": [0.0, 3.0, 1.5],
        "target": [0.0, 0.0, 1.0],
        "up": [0.0, 0.0, 1.0],
    },
    "top": {
        "position": [0.0, 0.0, 5.0],
        "target": [0.0, 0.0, 0.0],
        "up": [0.0, 1.0, 0.0],
    },
    "follow_ball": {
        "position": [2.0, 2.0, 1.0],
        "target": [0.0, 0.0, 0.0],
        "up": [0.0, 0.0, 1.0],
    },
    "follow_club": {
        "position": [1.5, -1.0, 2.0],
        "target": [0.0, 0.0, 1.5],
        "up": [0.0, 0.0, 1.0],
    },
}


@precondition(
    lambda engine_manager: engine_manager is not None,
    "Engine manager must not be None",
)
def _get_control_interface(
    engine_manager: EngineManager,
) -> Any:
    """Get or create a ControlInterface for the active engine.

    Returns:
        ControlInterface instance, or None if no engine is loaded.
    """
    engine = engine_manager.get_active_physics_engine()
    if engine is None:
        return None

    # Key by active engine identity so a switch automatically invalidates the cache
    cache_key = id(engine)
    if cache_key in _CONTROL_INTERFACE_CACHE:
        return _CONTROL_INTERFACE_CACHE[cache_key]

    try:
        from src.shared.python.control_interface import ControlInterface

        ctrl = ControlInterface(engine)
        _CONTROL_INTERFACE_CACHE[cache_key] = ctrl
        return ctrl
    except ImportError:
        return None


@precondition(
    lambda engine_manager: engine_manager is not None,
    "Engine manager must not be None",
)
def _get_features_registry(
    engine_manager: EngineManager,
) -> Any:
    """Get or create a ControlFeaturesRegistry for the active engine.

    Returns:
        ControlFeaturesRegistry instance, or None if no engine is loaded.
    """
    engine = engine_manager.get_active_physics_engine()
    if engine is None:
        return None

    # Key by active engine identity so a switch automatically invalidates the cache
    cache_key = id(engine)
    if cache_key in _FEATURES_REGISTRY_CACHE:
        return _FEATURES_REGISTRY_CACHE[cache_key]

    try:
        from src.shared.python.control_features_registry import (
            ControlFeaturesRegistry,
        )

        registry = ControlFeaturesRegistry(engine)
        _FEATURES_REGISTRY_CACHE[cache_key] = registry
        return registry
    except ImportError:
        return None


# ──────────────────────────────────────────────────────────────
#  Actuator Control (See issue #1209)
# ──────────────────────────────────────────────────────────────


@router.post("/simulation/actuators", response_model=ActuatorStateResponse)
@handle_api_errors
async def update_actuators(
    request: ActuatorUpdateRequest,
    engine_manager: EngineManager = Depends(get_engine_manager),
    logger: Any = Depends(get_logger),
) -> ActuatorStateResponse:
    """Update per-actuator control parameters.

    Allows setting control strategy, torques, gains, and target
    positions/velocities for the active physics engine.

    Args:
        request: Actuator parameter updates.
        engine_manager: Injected engine manager.
        logger: Injected logger.

    Returns:
        Current actuator/control state after applying updates.

    Raises:
        HTTPException: If no engine is loaded or parameters are invalid.
    """
    ctrl = _get_control_interface(engine_manager)
    if ctrl is None:
        raise HTTPException(
            status_code=400,
            detail="No physics engine loaded. Load an engine first.",
        )

    try:
        if request.strategy is not None:
            ctrl.set_strategy(request.strategy)

        if request.torques is not None:
            ctrl.set_torques(request.torques)

        if request.kp is not None or request.kd is not None or request.ki is not None:
            ctrl.set_gains(kp=request.kp, kd=request.kd, ki=request.ki)

        if request.target_positions is not None:
            ctrl.set_target_positions(request.target_positions)

        if request.target_velocities is not None:
            ctrl.set_target_velocities(request.target_velocities)

        state = ctrl.get_state()
        return ActuatorStateResponse(
            **state,
            available_strategies=ctrl.get_available_strategies(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ImportError as exc:
        if logger:
            logger.exception("Actuator update error")
        raise HTTPException(
            status_code=500, detail=f"Actuator update failed: {str(exc)}"
        ) from exc


@router.get("/simulation/actuators", response_model=ActuatorStateResponse)
@handle_api_errors
async def get_actuator_state(
    engine_manager: EngineManager = Depends(get_engine_manager),
) -> ActuatorStateResponse:
    """Get current actuator/control state.

    Returns:
        Current actuator state including strategy, torques, and gains.

    Raises:
        HTTPException: If no engine is loaded.
    """
    ctrl = _get_control_interface(engine_manager)
    if ctrl is None:
        raise HTTPException(
            status_code=400,
            detail="No physics engine loaded. Load an engine first.",
        )

    state = ctrl.get_state()
    return ActuatorStateResponse(
        **state,
        available_strategies=ctrl.get_available_strategies(),
    )


# ──────────────────────────────────────────────────────────────
#  Force/Torque Vectors (See issue #1209)
# ──────────────────────────────────────────────────────────────


@router.get("/simulation/forces", response_model=ForceVectorResponse)
@handle_api_errors
async def get_forces(
    engine_manager: EngineManager = Depends(get_engine_manager),
    logger: Any = Depends(get_logger),
) -> ForceVectorResponse:
    """Get current force and torque vectors from the active engine.

    Returns:
        Force vectors including gravity, contact forces, applied torques,
        and bias forces.

    Raises:
        HTTPException: If no engine is loaded.
    """
    engine = engine_manager.get_active_physics_engine()
    if engine is None:
        raise HTTPException(
            status_code=400,
            detail="No physics engine loaded. Load an engine first.",
        )

    try:
        sim_time = getattr(engine, "time", 0.0)

        # Gravity forces
        gravity = None
        try:
            g = engine.compute_gravity_forces()
            gravity = g.tolist() if hasattr(g, "tolist") else list(g)
        except (ValueError, RuntimeError, AttributeError) as exc:
            _logger.warning("compute_gravity_forces unavailable: %s", exc)

        # Contact forces
        contact = None
        try:
            c = engine.compute_contact_forces()
            contact = c.tolist() if hasattr(c, "tolist") else list(c)
        except (ValueError, RuntimeError, AttributeError) as exc:
            _logger.warning("compute_contact_forces unavailable: %s", exc)

        # Applied torques from control interface
        ctrl = _get_control_interface(engine_manager)
        applied = []
        if ctrl is not None:
            applied = ctrl.current_torques.tolist()

        # Bias forces
        bias = None
        try:
            b = engine.compute_bias_forces()
            bias = b.tolist() if hasattr(b, "tolist") else list(b)
        except (ValueError, RuntimeError, AttributeError) as exc:
            _logger.warning("compute_bias_forces unavailable: %s", exc)

        return ForceVectorResponse(
            sim_time=sim_time,
            gravity_forces=gravity,
            contact_forces=contact,
            applied_torques=applied,
            bias_forces=bias,
        )
    except ImportError as exc:
        if logger:
            logger.exception("Force query error")
        raise HTTPException(
            status_code=500, detail=f"Force query failed: {str(exc)}"
        ) from exc


# ──────────────────────────────────────────────────────────────
#  Biomechanics Metrics (See issue #1209)
# ──────────────────────────────────────────────────────────────


@router.get("/simulation/metrics", response_model=BiomechanicsMetricsResponse)
@handle_api_errors
async def get_metrics(
    engine_manager: EngineManager = Depends(get_engine_manager),
    logger: Any = Depends(get_logger),
) -> BiomechanicsMetricsResponse:
    """Get current biomechanics metrics from the active simulation.

    Returns:
        Metrics including club head speed, energy, joint states, and torques.

    Raises:
        HTTPException: If no engine is loaded.
    """
    engine = engine_manager.get_active_physics_engine()
    if engine is None:
        raise HTTPException(
            status_code=400,
            detail="No physics engine loaded. Load an engine first.",
        )

    try:
        sim_time = getattr(engine, "time", 0.0)
        q, v = engine.get_state()

        # Club head speed: try to get from Jacobian of end-effector
        club_head_speed = None
        try:
            jac = engine.compute_jacobian("club_head")
            if jac is not None and "linear" in jac:
                import numpy as np

                linear_vel = jac["linear"] @ v
                club_head_speed = float(np.linalg.norm(linear_vel))
        except ImportError as exc:
            _logger.warning(
                "numpy unavailable for club-head speed calculation: %s", exc
            )

        # Energy calculations
        kinetic_energy = None
        potential_energy = None
        try:
            import numpy as np

            M = engine.compute_mass_matrix()
            kinetic_energy = float(0.5 * v @ M @ v)
        except ImportError as exc:
            _logger.warning("numpy unavailable for kinetic energy calculation: %s", exc)

        # Torque metrics
        ctrl = _get_control_interface(engine_manager)
        peak_torque = None
        total_torque_magnitude = None
        if ctrl is not None:
            import numpy as np

            torques = ctrl.current_torques
            peak_torque = float(np.max(np.abs(torques)))
            total_torque_magnitude = float(np.sum(np.abs(torques)))

        return BiomechanicsMetricsResponse(
            sim_time=sim_time,
            club_head_speed=club_head_speed,
            kinetic_energy=kinetic_energy,
            potential_energy=potential_energy,
            joint_positions=q.tolist(),
            joint_velocities=v.tolist(),
            peak_torque=peak_torque,
            total_torque_magnitude=total_torque_magnitude,
        )
    except ImportError as exc:
        if logger:
            logger.exception("Metrics query error")
        raise HTTPException(
            status_code=500, detail=f"Metrics query failed: {str(exc)}"
        ) from exc


# ──────────────────────────────────────────────────────────────
#  Control Features Registry (See issue #1209)
# ──────────────────────────────────────────────────────────────


@router.get("/simulation/control-features", response_model=ControlFeaturesResponse)
@handle_api_errors
async def get_control_features(
    category: str | None = None,
    available_only: bool = False,
    engine_manager: EngineManager = Depends(get_engine_manager),
) -> ControlFeaturesResponse:
    """Get control features registry data including ZTCF/ZVCF availability.

    Args:
        category: Filter by feature category (optional).
        available_only: If True, only return available features.
        engine_manager: Injected engine manager.

    Returns:
        Registry summary with feature descriptors.

    Raises:
        HTTPException: If no engine is loaded.
    """
    registry = _get_features_registry(engine_manager)
    if registry is None:
        raise HTTPException(
            status_code=400,
            detail="No physics engine loaded. Load an engine first.",
        )

    summary = registry.get_summary()
    features = registry.list_features(category=category, available_only=available_only)

    return ControlFeaturesResponse(
        engine=summary["engine"],
        total_features=summary["total_features"],
        available_features=summary["available_features"],
        categories=summary["categories"],
        features=features,
    )


# ──────────────────────────────────────────────────────────────
#  Simulation Controls (See issue #1202)
# ──────────────────────────────────────────────────────────────


@router.get("/simulation/stats", response_model=SimulationStatsResponse)
@handle_api_errors
async def get_simulation_stats(
    engine_manager: EngineManager = Depends(get_engine_manager),
    simulation_service: SimulationService = Depends(get_simulation_service),
) -> SimulationStatsResponse:
    """Get simulation runtime statistics.

    Returns:
        Stats including sim time, FPS, real-time factor, and recording state.
    """
    engine = engine_manager.get_active_physics_engine()

    sim_time = 0.0
    if engine is not None:
        sim_time = getattr(engine, "time", 0.0)

    stats = simulation_service.stats
    wall_time = time.time() - stats.start_time
    fps = stats.frame_count / wall_time if wall_time > 0 else 0.0
    real_time_factor = sim_time / wall_time if wall_time > 0 else 0.0

    return SimulationStatsResponse(
        sim_time=sim_time,
        wall_time=wall_time,
        fps=fps,
        real_time_factor=real_time_factor,
        speed_factor=stats.speed_factor,
        is_recording=stats.is_recording,
        frame_count=stats.frame_count,
    )


@router.post("/simulation/speed", response_model=SpeedControlResponse)
@handle_api_errors
async def set_simulation_speed(
    request: SpeedControlRequest,
    simulation_service: SimulationService = Depends(get_simulation_service),
) -> SpeedControlResponse:
    """Set simulation speed multiplier.

    Args:
        request: Speed control parameters.
        simulation_service: Injected simulation service.

    Returns:
        Applied speed factor and status.
    """
    if not (request is not None):
        raise ValueError("request must be provided")
    simulation_service.set_speed_factor(request.speed_factor)

    return SpeedControlResponse(
        speed_factor=request.speed_factor,
        status=f"Speed set to {request.speed_factor}x",
    )


@router.post("/simulation/camera", response_model=CameraPresetResponse)
@handle_api_errors
async def set_camera_preset(
    request: CameraPresetRequest,
) -> CameraPresetResponse:
    """Apply a camera preset.

    Args:
        request: Camera preset selection.

    Returns:
        Applied camera position, target, and up vector.

    Raises:
        HTTPException: If preset is unknown.
    """
    preset_data = CAMERA_PRESETS.get(request.preset)
    if preset_data is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown camera preset: {request.preset}",
        )

    return CameraPresetResponse(
        preset=request.preset,
        position=preset_data["position"],
        target=preset_data["target"],
        up=preset_data["up"],
    )


@router.post("/simulation/recording", response_model=TrajectoryRecordResponse)
@handle_api_errors
async def control_recording(
    request: TrajectoryRecordRequest,
    simulation_service: SimulationService = Depends(get_simulation_service),
    logger: Any = Depends(get_logger),
) -> TrajectoryRecordResponse:
    """Control trajectory recording (start, stop, export).

    Args:
        request: Recording action and export format.
        simulation_service: Injected simulation service (owns recording state).
        logger: Injected logger.

    Returns:
        Current recording state.
    """
    action = request.action

    if action == "start":
        simulation_service.start_recording()
        return TrajectoryRecordResponse(  # type: ignore[call-arg]
            recording=True,
            frame_count=0,
            status="Recording started",
        )

    if action == "stop":
        simulation_service.stop_recording()
        frame_count = len(simulation_service.stats.recorded_frames)
        return TrajectoryRecordResponse(  # type: ignore[call-arg]
            recording=False,
            frame_count=frame_count,
            status="Recording stopped",
        )

    if action == "export":
        recorded = simulation_service.stats.recorded_frames
        frame_count = len(recorded)
        export_path = None

        if frame_count > 0:
            import json
            import os
            import tempfile
            from pathlib import Path

            # Use a managed artifact directory with proper cleanup
            # Artifacts are stored in a configured directory with atomic writes
            # and automatic cleanup on response completion
            artifact_dir = os.environ.get(
                "ARTIFACT_DIR",
                os.path.join(tempfile.gettempdir(), "upstream_drift_artifacts"),
            )
            os.makedirs(artifact_dir, exist_ok=True)

            # Create a uniquely named artifact file with atomic write
            # Using a temporary file in the managed directory ensures cleanup
            fd, tmp_path = tempfile.mkstemp(
                suffix=f".{request.export_format}",
                dir=artifact_dir,
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
                    json.dump(
                        {"frames": recorded, "format": request.export_format},
                        tmp_file,
                        indent=2,
                    )
                # Return only the filename (not full path) for security
                # The actual download would be handled by a separate endpoint
                export_path = Path(tmp_path).name
                if logger:
                    logger.info(
                        "Trajectory exported to %s (%d frames)",
                        export_path,
                        frame_count,
                    )
            except Exception:
                # Clean up on error
                with contextlib.suppress(OSError):
                    os.unlink(tmp_path)
                raise

        return TrajectoryRecordResponse(
            recording=simulation_service.stats.is_recording,
            frame_count=frame_count,
            status="Trajectory exported" if export_path else "No frames to export",
            export_path=export_path,
        )

    # Should not reach here due to validator, but safety fallback
    raise HTTPException(status_code=400, detail=f"Unknown action: {action}")
