"""Force/torque vector overlay routes.

Provides endpoints for streaming force/torque visualization data
from the active simulation engine. Supports filtering by body,
force type, and magnitude-based color coding.

See issue #1199

All dependencies are injected via FastAPI's Depends() mechanism.
No module-level mutable state.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_engine_manager, get_logger
from ..models.requests import ForceOverlayRequest
from ..models.responses import (
    ForceOverlayResponse,
    ForceVector3D,
)

if TYPE_CHECKING:
    from src.shared.python.engine_core.engine_manager import EngineManager

router = APIRouter()

# Color mapping for force types. See issue #1199
FORCE_TYPE_COLORS: dict[str, list[float]] = {
    "applied": [1.0, 0.2, 0.2, 1.0],  # Red
    "gravity": [0.2, 0.6, 1.0, 1.0],  # Blue
    "contact": [0.2, 1.0, 0.2, 1.0],  # Green
    "bias": [1.0, 0.8, 0.2, 1.0],  # Yellow
}


def _magnitude_to_color(magnitude: float, max_magnitude: float) -> list[float]:
    """Map a force magnitude to a heat-map color (blue -> red).

    Args:
        magnitude: The force magnitude.
        max_magnitude: Maximum expected magnitude for normalization.

    Returns:
        RGBA color list.
    """
    if max_magnitude <= 0:
        return [0.5, 0.5, 0.5, 1.0]
    t = min(magnitude / max_magnitude, 1.0)
    # Blue (cold) to red (hot) via green
    r = min(2.0 * t, 1.0)
    g = min(2.0 * (1.0 - t), 1.0)
    b = max(1.0 - 2.0 * t, 0.0)
    return [r, g, b, 1.0]


def _build_force_vectors(
    engine_manager: EngineManager,
    config: ForceOverlayRequest,
) -> list[ForceVector3D]:
    """Build force vector list from engine state.

    Queries the active engine for force/torque data and converts
    it into renderable 3D vectors.

    Args:
        engine_manager: Active engine manager.
        config: Overlay configuration.

    Returns:
        List of force vectors for rendering.
    """
    vectors: list[ForceVector3D] = []

    try:
        engine = engine_manager.get_active_engine()
    except (AttributeError, RuntimeError):
        # No active engine, return synthetic demo vectors
        return _build_demo_vectors(config)

    if engine is None:
        return _build_demo_vectors(config)

    # Try to get force data from engine state
    try:
        state = engine.get_state() if hasattr(engine, "get_state") else {}
    except (ValueError, RuntimeError, AttributeError):
        state = {}

    positions = state.get("positions", [])
    torques = state.get("torques", [])
    joint_names = []
    if hasattr(engine, "joint_names"):
        joint_names = engine.joint_names
    elif hasattr(engine, "get_joint_names"):
        joint_names = engine.get_joint_names()

    n_joints = len(torques) if torques else len(positions) if positions else 0

    # Generate names if not available
    if not joint_names:
        joint_names = [f"joint_{i}" for i in range(n_joints)]

    # Build applied torque vectors
    if "applied" in config.force_types or "all" in config.force_types:
        for i in range(min(n_joints, len(torques) if torques else 0)):
            torque_val = torques[i] if torques else 0.0
            if abs(torque_val) < 1e-6:
                continue

            body_name = joint_names[i] if i < len(joint_names) else f"joint_{i}"
            if config.body_filter and body_name not in config.body_filter:
                continue

            # Position along a vertical chain (simplified)
            y_pos = 0.5 + i * 0.3
            direction = [0.0, 0.0, 1.0] if torque_val > 0 else [0.0, 0.0, -1.0]

            vectors.append(
                ForceVector3D(
                    body_name=body_name,
                    force_type="applied",
                    origin=[0.0, y_pos, 0.0],
                    direction=direction,
                    magnitude=abs(torque_val),
                    color=FORCE_TYPE_COLORS["applied"],
                    label=f"{torque_val:.1f} N*m" if config.show_labels else None,
                )
            )

    # Build gravity vectors
    if "gravity" in config.force_types or "all" in config.force_types:
        for i in range(n_joints):
            body_name = joint_names[i] if i < len(joint_names) else f"joint_{i}"
            if config.body_filter and body_name not in config.body_filter:
                continue

            y_pos = 0.5 + i * 0.3
            # Gravity is always downward
            gravity_mag = 9.81 * (0.5 + 0.1 * i)  # Approximate per-body mass * g
            vectors.append(
                ForceVector3D(
                    body_name=body_name,
                    force_type="gravity",
                    origin=[0.0, y_pos, 0.0],
                    direction=[0.0, -1.0, 0.0],
                    magnitude=gravity_mag,
                    color=FORCE_TYPE_COLORS["gravity"],
                    label=f"{gravity_mag:.1f} N" if config.show_labels else None,
                )
            )

    # Apply magnitude-based coloring if enabled
    if config.color_by_magnitude and vectors:
        max_mag = max(v.magnitude for v in vectors)
        for v in vectors:
            v.color = _magnitude_to_color(v.magnitude, max_mag)

    return vectors


def _build_demo_vectors(config: ForceOverlayRequest) -> list[ForceVector3D]:
    """Build demo force vectors when no engine is active.

    Args:
        config: Overlay configuration.

    Returns:
        List of demo force vectors.
    """
    demo_bodies = ["torso", "upper_arm", "forearm", "hand", "club"]
    vectors: list[ForceVector3D] = []

    for i, body in enumerate(demo_bodies):
        if config.body_filter and body not in config.body_filter:
            continue

        y_pos = 0.5 + i * 0.3

        if "applied" in config.force_types or "all" in config.force_types:
            mag = 10.0 + i * 5.0
            vectors.append(
                ForceVector3D(
                    body_name=body,
                    force_type="applied",
                    origin=[0.0, y_pos, 0.0],
                    direction=[math.sin(i * 0.5), 0.0, math.cos(i * 0.5)],
                    magnitude=mag,
                    color=FORCE_TYPE_COLORS["applied"],
                    label=f"{mag:.1f} N*m" if config.show_labels else None,
                )
            )

        if "gravity" in config.force_types or "all" in config.force_types:
            grav_mag = 9.81 * (1.0 + 0.2 * i)
            vectors.append(
                ForceVector3D(
                    body_name=body,
                    force_type="gravity",
                    origin=[0.0, y_pos, 0.0],
                    direction=[0.0, -1.0, 0.0],
                    magnitude=grav_mag,
                    color=FORCE_TYPE_COLORS["gravity"],
                    label=f"{grav_mag:.1f} N" if config.show_labels else None,
                )
            )

    return vectors


@router.get(
    "/simulation/forces",
    response_model=ForceOverlayResponse,
)
async def get_force_overlays(
    force_types: str = "applied",
    color_by_magnitude: bool = True,
    body_filter: str | None = None,
    show_labels: bool = False,
    scale_factor: float = 0.01,
    engine_manager: Any = Depends(get_engine_manager),
    logger: Any = Depends(get_logger),
) -> ForceOverlayResponse:
    """Get current force/torque vectors for 3D overlay rendering.

    Queries the active simulation engine for force data and returns
    structured vector data suitable for Three.js ArrowHelper rendering.

    Args:
        force_types: Comma-separated force types to include.
        color_by_magnitude: Whether to color-code by magnitude.
        body_filter: Comma-separated body names to filter (None = all).
        show_labels: Whether to include text labels.
        scale_factor: Arrow length scaling factor.
        engine_manager: Injected engine manager.
        logger: Injected logger.

    Returns:
        Force overlay data with vectors and metadata.
    """
    try:
        config = ForceOverlayRequest(
            enabled=True,
            force_types=force_types.split(","),
            color_by_magnitude=color_by_magnitude,
            body_filter=body_filter.split(",") if body_filter else None,
            show_labels=show_labels,
            scale_factor=scale_factor,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        vectors = _build_force_vectors(engine_manager, config)

        total_force = sum(v.magnitude for v in vectors if v.force_type != "bias")
        total_torque = sum(v.magnitude for v in vectors if v.force_type == "applied")

        # Get simulation time
        sim_time = 0.0
        try:
            active = engine_manager.get_active_engine()
            if active and hasattr(active, "get_state"):
                state = active.get_state()
                sim_time = state.get("time", 0.0)
        except (ValueError, RuntimeError, AttributeError):
            pass

        return ForceOverlayResponse(
            sim_time=sim_time,
            vectors=vectors,
            total_force_magnitude=total_force,
            total_torque_magnitude=total_torque,
            overlay_config={
                "force_types": config.force_types,
                "color_by_magnitude": config.color_by_magnitude,
                "scale_factor": config.scale_factor,
                "body_filter": config.body_filter,
            },
        )
    except (ValueError, RuntimeError, AttributeError) as exc:
        if logger:
            logger.error("Error building force overlays: %s", exc)
        raise HTTPException(
            status_code=500, detail=f"Force overlay error: {str(exc)}"
        ) from exc


@router.post(
    "/simulation/forces/config",
    response_model=ForceOverlayResponse,
)
async def update_force_overlay_config(
    config: ForceOverlayRequest,
    engine_manager: Any = Depends(get_engine_manager),
    logger: Any = Depends(get_logger),
) -> ForceOverlayResponse:
    """Update force overlay configuration and return current vectors.

    Allows the frontend to POST a complete overlay configuration
    and receive the matching vector data in response.

    Args:
        config: Force overlay configuration.
        engine_manager: Injected engine manager.
        logger: Injected logger.

    Returns:
        Updated force overlay data.
    """
    try:
        vectors = _build_force_vectors(engine_manager, config)

        total_force = sum(v.magnitude for v in vectors if v.force_type != "bias")
        total_torque = sum(v.magnitude for v in vectors if v.force_type == "applied")

        sim_time = 0.0
        try:
            active = engine_manager.get_active_engine()
            if active and hasattr(active, "get_state"):
                state = active.get_state()
                sim_time = state.get("time", 0.0)
        except (ValueError, RuntimeError, AttributeError):
            pass

        return ForceOverlayResponse(
            sim_time=sim_time,
            vectors=vectors,
            total_force_magnitude=total_force,
            total_torque_magnitude=total_torque,
            overlay_config={
                "force_types": config.force_types,
                "color_by_magnitude": config.color_by_magnitude,
                "scale_factor": config.scale_factor,
                "body_filter": config.body_filter,
                "show_labels": config.show_labels,
            },
        )
    except (ValueError, RuntimeError, AttributeError) as exc:
        if logger:
            logger.error("Error updating force overlay config: %s", exc)
        raise HTTPException(
            status_code=500, detail=f"Force overlay error: {str(exc)}"
        ) from exc
