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

from fastapi import APIRouter, Depends

from src.api.middleware.error_handler import handle_api_errors
from src.shared.python.core.constants import GRAVITY
from src.shared.python.core.contracts import precondition

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


def _extract_engine_state(engine_manager: EngineManager) -> tuple[Any, dict]:
    """Extract the active engine and its current state.

    Args:
        engine_manager: The engine manager instance.

    Returns:
        Tuple of (engine instance or None, state dict).
    """
    try:
        engine = engine_manager.get_active_engine()  # type: ignore[attr-defined]
    except (AttributeError, RuntimeError):
        return None, {}

    if engine is None:
        return None, {}

    try:
        state = engine.get_state() if hasattr(engine, "get_state") else {}
    except (ValueError, RuntimeError, AttributeError):
        state = {}

    return engine, state


def _resolve_joint_names(engine: Any, n_joints: int) -> list[str]:
    """Resolve joint names from the engine, falling back to generic names.

    Args:
        engine: The active physics engine instance.
        n_joints: Number of joints to generate names for.

    Returns:
        List of joint name strings.
    """
    joint_names: list[str] = []
    if hasattr(engine, "joint_names"):
        joint_names = engine.joint_names
    elif hasattr(engine, "get_joint_names"):
        joint_names = engine.get_joint_names()
    if not joint_names:
        joint_names = [f"joint_{i}" for i in range(n_joints)]
    return joint_names


def _should_include_force_type(config: ForceOverlayRequest, force_type: str) -> bool:
    """Check whether a force type should be included based on the config.

    Args:
        config: The overlay request configuration.
        force_type: Force type string to check (e.g. "applied", "gravity").

    Returns:
        True if the force type is included in the config.
    """
    return force_type in config.force_types or "all" in config.force_types


def _resolve_body_name(joint_names: list[str], index: int) -> str:
    """Resolve a body name from joint names by index, with fallback.

    Args:
        joint_names: List of known joint names.
        index: Joint index to look up.

    Returns:
        The joint name at the given index, or a generic "joint_N" name.
    """
    if index < len(joint_names):
        return joint_names[index]
    return f"joint_{index}"


def _is_filtered_out(config: ForceOverlayRequest, body_name: str) -> bool:
    """Check whether a body should be excluded by the body filter.

    Args:
        config: The overlay request configuration.
        body_name: Name of the body to check.

    Returns:
        True if the body is filtered out (not in the filter list).
    """
    return bool(config.body_filter and body_name not in config.body_filter)


def _build_applied_torque_vectors(
    config: ForceOverlayRequest,
    torques: list,
    joint_names: list[str],
    n_joints: int,
) -> list[ForceVector3D]:
    """Build 3D vectors for applied joint torques.

    Args:
        config: Overlay request configuration.
        torques: List of torque values per joint.
        joint_names: List of joint name strings.
        n_joints: Number of joints.

    Returns:
        List of ForceVector3D for non-zero applied torques.
    """
    if not _should_include_force_type(config, "applied"):
        return []

    torque_count = len(torques) if torques else 0
    vectors: list[ForceVector3D] = []
    for i in range(min(n_joints, torque_count)):
        torque_val = torques[i]
        if abs(torque_val) < 1e-6:
            continue

        body_name = _resolve_body_name(joint_names, i)
        if _is_filtered_out(config, body_name):
            continue

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
    return vectors


def _build_gravity_vectors(
    config: ForceOverlayRequest,
    joint_names: list[str],
    n_joints: int,
) -> list[ForceVector3D]:
    """Build 3D vectors for gravity forces on each joint.

    Args:
        config: Overlay request configuration.
        joint_names: List of joint name strings.
        n_joints: Number of joints.

    Returns:
        List of ForceVector3D for downward gravity forces.
    """
    if not _should_include_force_type(config, "gravity"):
        return []

    vectors: list[ForceVector3D] = []
    for i in range(n_joints):
        body_name = _resolve_body_name(joint_names, i)
        if _is_filtered_out(config, body_name):
            continue

        y_pos = 0.5 + i * 0.3
        gravity_mag = GRAVITY * (0.5 + 0.1 * i)
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
    return vectors


def _apply_magnitude_coloring(vectors: list[ForceVector3D]) -> None:
    """Re-color vectors using a heat-map based on magnitude.

    Args:
        vectors: List of force vectors to re-color in place.
    """
    if not vectors:
        return
    max_mag = max(v.magnitude for v in vectors)
    for v in vectors:
        v.color = _magnitude_to_color(v.magnitude, max_mag)


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
    engine, state = _extract_engine_state(engine_manager)
    if engine is None:
        return _build_demo_vectors(config)

    positions = state.get("positions", [])
    torques = state.get("torques", [])
    if torques:
        n_joints = len(torques)
    elif positions:
        n_joints = len(positions)
    else:
        n_joints = 0
    joint_names = _resolve_joint_names(engine, n_joints)

    vectors: list[ForceVector3D] = []
    vectors.extend(
        _build_applied_torque_vectors(config, torques, joint_names, n_joints)
    )
    vectors.extend(_build_gravity_vectors(config, joint_names, n_joints))

    if config.color_by_magnitude:
        _apply_magnitude_coloring(vectors)

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
            grav_mag = GRAVITY * (1.0 + 0.2 * i)
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


def _get_sim_time(engine_manager: EngineManager) -> float:
    """Get current simulation time from the engine manager.

    Breaks apart the chained engine_manager.get_active_engine().get_state()
    train wreck into a safe helper.

    Args:
        engine_manager: The engine manager to query.

    Returns:
        Current simulation time, or 0.0 if unavailable.
    """
    try:
        active = engine_manager.get_active_engine()  # type: ignore[attr-defined]
        if active is None or not hasattr(active, "get_state"):
            return 0.0
        state = active.get_state()
        return float(state.get("time", 0.0))
    except (ValueError, RuntimeError, AttributeError):
        return 0.0


@router.get(
    "/simulation/forces",
    response_model=ForceOverlayResponse,
)
@precondition(
    lambda force_types="applied",
    color_by_magnitude=True,
    body_filter=None,
    show_labels=False,
    scale_factor=0.01,
    engine_manager=None,
    logger=None: scale_factor > 0 and len(force_types.strip()) > 0,
    "Scale factor must be positive and force_types must be non-empty",
)
@handle_api_errors
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
    config = ForceOverlayRequest(
        enabled=True,
        force_types=force_types.split(","),
        color_by_magnitude=color_by_magnitude,
        body_filter=body_filter.split(",") if body_filter else None,
        show_labels=show_labels,
        scale_factor=scale_factor,
    )

    vectors = _build_force_vectors(engine_manager, config)

    total_force = sum(v.magnitude for v in vectors if v.force_type != "bias")
    total_torque = sum(v.magnitude for v in vectors if v.force_type == "applied")

    sim_time = _get_sim_time(engine_manager)

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


@router.post(
    "/simulation/forces/config",
    response_model=ForceOverlayResponse,
)
@precondition(
    lambda config, engine_manager=None, logger=None: config.scale_factor > 0,
    "Scale factor must be positive",
)
@handle_api_errors
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
    vectors = _build_force_vectors(engine_manager, config)

    total_force = sum(v.magnitude for v in vectors if v.force_type != "bias")
    total_torque = sum(v.magnitude for v in vectors if v.force_type == "applied")

    sim_time = _get_sim_time(engine_manager)

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
