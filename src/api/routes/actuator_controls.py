"""Per-actuator control routes.

Provides endpoints for querying actuator state and sending
per-actuator commands with multiple control types (constant,
polynomial, PD gains, trajectory).

See issue #1198

All dependencies are injected via FastAPI's Depends() mechanism.
No module-level mutable state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_engine_manager, get_logger
from ..models.requests import (
    ActuatorBatchCommandRequest,
    ActuatorCommandRequest,
)
from ..models.responses import (
    ActuatorCommandResponse,
    ActuatorInfo,
    ActuatorPanelResponse,
)

if TYPE_CHECKING:
    from src.shared.python.engine_core.engine_manager import EngineManager

router = APIRouter()


def _get_actuator_info(engine_manager: EngineManager) -> list[ActuatorInfo]:
    """Extract actuator information from the active engine.

    Args:
        engine_manager: Active engine manager.

    Returns:
        List of actuator descriptors.
    """
    actuators: list[ActuatorInfo] = []
    manager_any = cast(Any, engine_manager)

    try:
        engine = manager_any.get_active_engine()
    except (AttributeError, RuntimeError):
        return _demo_actuators()

    if engine is None:
        return _demo_actuators()

    # Try to query engine for joint/actuator info
    joint_names: list[str] = []
    if hasattr(engine, "joint_names"):
        joint_names = list(engine.joint_names)
    elif hasattr(engine, "get_joint_names"):
        joint_names = list(engine.get_joint_names())

    n_joints = len(joint_names) if joint_names else 0

    # Get current state
    try:
        state = engine.get_state() if hasattr(engine, "get_state") else {}
    except (ValueError, RuntimeError, AttributeError):
        state = {}

    torques = state.get("torques", [0.0] * n_joints)

    # Get limits if available
    for i in range(n_joints):
        name = joint_names[i] if i < len(joint_names) else f"joint_{i}"
        current_torque = torques[i] if i < len(torques) else 0.0

        # Try to get limits from engine
        min_val = -100.0
        max_val = 100.0
        if hasattr(engine, "get_joint_limits"):
            try:
                limits = engine.get_joint_limits()
                if i < len(limits):
                    min_val, max_val = limits[i]
            except (ValueError, RuntimeError, AttributeError):
                pass

        actuators.append(
            ActuatorInfo(
                index=i,
                name=name,
                control_type="constant",
                value=current_torque,
                min_value=min_val,
                max_value=max_val,
                units="N*m",
                joint_type="revolute",
            )
        )

    return actuators if actuators else _demo_actuators()


def _demo_actuators() -> list[ActuatorInfo]:
    """Return demo actuator data when no engine is active.

    Returns:
        List of demo actuator descriptors.
    """
    demo_joints = [
        ("hip_rotation", -3.14, 3.14, "revolute"),
        ("shoulder_flexion", -2.0, 2.0, "revolute"),
        ("shoulder_rotation", -1.5, 1.5, "revolute"),
        ("elbow_flexion", 0.0, 2.5, "revolute"),
        ("wrist_flexion", -1.0, 1.0, "revolute"),
        ("wrist_deviation", -0.5, 0.5, "revolute"),
    ]
    return [
        ActuatorInfo(
            index=i,
            name=name,
            control_type="constant",
            value=0.0,
            min_value=min_v,
            max_value=max_v,
            units="N*m",
            joint_type=jtype,
        )
        for i, (name, min_v, max_v, jtype) in enumerate(demo_joints)
    ]


@router.get(
    "/simulation/actuators",
    response_model=ActuatorPanelResponse,
)
async def get_actuator_panel(
    engine_manager: Any = Depends(get_engine_manager),
    logger: Any = Depends(get_logger),
) -> ActuatorPanelResponse:
    """Get the current actuator panel state.

    Returns per-actuator descriptors with current values, limits,
    and available control types for the frontend slider panel.

    Args:
        engine_manager: Injected engine manager.
        logger: Injected logger.

    Returns:
        Actuator panel state.
    """
    try:
        actuators = _get_actuator_info(engine_manager)

        engine_name = "none"
        try:
            active = cast(Any, engine_manager).get_active_engine()
            if active and hasattr(active, "engine_type"):
                engine_name = str(active.engine_type)
            elif active:
                engine_name = type(active).__name__
        except (RuntimeError, ValueError, AttributeError):
            pass

        return ActuatorPanelResponse(
            n_actuators=len(actuators),
            actuators=actuators,
            available_control_types=[
                "constant",
                "polynomial",
                "pd_gains",
                "trajectory",
            ],
            engine_name=engine_name,
        )
    except (RuntimeError, TypeError, AttributeError) as exc:
        if logger:
            logger.error("Error getting actuator panel: %s", exc)
        raise HTTPException(
            status_code=500, detail=f"Actuator panel error: {str(exc)}"
        ) from exc


@router.post(
    "/simulation/actuators",
    response_model=ActuatorCommandResponse,
)
async def send_actuator_command(
    command: ActuatorCommandRequest,
    engine_manager: Any = Depends(get_engine_manager),
    logger: Any = Depends(get_logger),
) -> ActuatorCommandResponse:
    """Send a command to a single actuator.

    Applies the specified value using the given control type.
    Values are clamped to actuator limits.

    Args:
        command: Actuator command.
        engine_manager: Injected engine manager.
        logger: Injected logger.

    Returns:
        Command acknowledgment with applied value.
    """
    try:
        actuators = _get_actuator_info(engine_manager)

        if command.actuator_index >= len(actuators):
            raise HTTPException(
                status_code=400,
                detail=f"Actuator index {command.actuator_index} out of range "
                f"(0..{len(actuators) - 1})",
            )

        actuator = actuators[command.actuator_index]
        clamped = False
        applied_value = command.value

        # Clamp to limits
        if applied_value < actuator.min_value:
            applied_value = actuator.min_value
            clamped = True
        elif applied_value > actuator.max_value:
            applied_value = actuator.max_value
            clamped = True

        # Apply to engine if available
        try:
            engine = engine_manager.get_active_engine()
            if engine and hasattr(engine, "set_control"):
                engine.set_control(command.actuator_index, applied_value)
            elif engine and hasattr(engine, "apply_torque"):
                engine.apply_torque(command.actuator_index, applied_value)
        except (ValueError, RuntimeError, AttributeError) as engine_err:
            if logger:
                logger.warning(
                    "Could not apply actuator command to engine: %s",
                    engine_err,
                )

        return ActuatorCommandResponse(
            actuator_index=command.actuator_index,
            applied_value=applied_value,
            control_type=command.control_type,
            status="ok",
            clamped=clamped,
        )
    except HTTPException:
        raise
    except (ValueError, RuntimeError, AttributeError) as exc:
        if logger:
            logger.error("Error sending actuator command: %s", exc)
        raise HTTPException(
            status_code=500, detail=f"Actuator command error: {str(exc)}"
        ) from exc


@router.post(
    "/simulation/actuators/batch",
    response_model=list[ActuatorCommandResponse],
)
async def send_actuator_batch(
    batch: ActuatorBatchCommandRequest,
    engine_manager: Any = Depends(get_engine_manager),
    logger: Any = Depends(get_logger),
) -> list[ActuatorCommandResponse]:
    """Send commands to multiple actuators simultaneously.

    Applies all commands in order and returns per-actuator results.

    Args:
        batch: Batch of actuator commands.
        engine_manager: Injected engine manager.
        logger: Injected logger.

    Returns:
        List of command acknowledgments.
    """
    results: list[ActuatorCommandResponse] = []
    actuators = _get_actuator_info(engine_manager)

    for cmd in batch.commands:
        if cmd.actuator_index >= len(actuators):
            results.append(
                ActuatorCommandResponse(
                    actuator_index=cmd.actuator_index,
                    applied_value=0.0,
                    control_type=cmd.control_type,
                    status=f"error: index {cmd.actuator_index} out of range",
                    clamped=False,
                )
            )
            continue

        actuator = actuators[cmd.actuator_index]
        clamped = False
        applied_value = cmd.value

        if applied_value < actuator.min_value:
            applied_value = actuator.min_value
            clamped = True
        elif applied_value > actuator.max_value:
            applied_value = actuator.max_value
            clamped = True

        try:
            engine = engine_manager.get_active_engine()
            if engine and hasattr(engine, "set_control"):
                engine.set_control(cmd.actuator_index, applied_value)
        except (ValueError, RuntimeError, AttributeError):
            pass

        results.append(
            ActuatorCommandResponse(
                actuator_index=cmd.actuator_index,
                applied_value=applied_value,
                control_type=cmd.control_type,
                status="ok",
                clamped=clamped,
            )
        )

    return results
