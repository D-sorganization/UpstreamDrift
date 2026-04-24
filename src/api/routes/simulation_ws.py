"""WebSocket routes for real-time simulation streaming."""

import asyncio
import contextlib
from typing import Any

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from src.shared.python.core.contracts import require
from src.shared.python.engine_core.engine_registry import EngineType

router = APIRouter()


class RunSummary(BaseModel):
    """Post-simulation summary returned after a run completes.

    Postcondition: all numeric fields are non-negative.
    """

    engine: str = Field(..., description="Physics engine used for this run")
    duration_s: float = Field(..., description="Simulated duration in seconds")
    steps: int = Field(..., description="Total simulation steps executed")
    max_torques: list[float] = Field(
        default_factory=list,
        description="Peak joint torques observed (N·m), one per DOF",
    )
    energy: float = Field(
        default=0.0, description="Total mechanical energy at end of run (J)"
    )
    trajectory: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Sampled trajectory states [{time, q, v}, …]",
    )
    next_steps: list[str] = Field(
        default_factory=list,
        description="Suggested follow-up actions based on the run",
    )


def _engine_type_from_str(name: str) -> EngineType:
    """Resolve an engine type string to EngineType, accepting any case.

    Args:
        name: Engine identifier (e.g. 'mujoco', 'MUJOCO', 'MuJoCo').

    Returns:
        Matching EngineType enum member.

    Raises:
        ValueError: If the name does not match any registered engine.
    """
    return EngineType(name.lower())


def _apply_initial_state(engine: object, state_dict: dict[str, Any]) -> None:
    """Apply an initial state dict to the engine using the (q, v) contract.

    Engines expose ``set_state(q: np.ndarray, v: np.ndarray)``.  The WebSocket
    client sends ``{"q": [...], "v": [...]}``; this helper converts and
    dispatches correctly.

    Args:
        engine: The active physics engine instance.
        state_dict: Dict with optional 'q' and 'v' lists.
    """
    if not hasattr(engine, "set_state"):
        return
    q = np.array(state_dict.get("q", []), dtype=float)
    v = np.array(state_dict.get("v", []), dtype=float)
    engine.set_state(q, v)


def _engine_state_to_dict(engine: object) -> dict[str, Any]:
    """Serialise engine state to a JSON-safe dict with 'q' and 'v' lists.

    ``engine.get_state()`` returns ``(np.ndarray, np.ndarray)``; raw numpy
    arrays are not JSON-serialisable, so we convert them to plain Python lists.

    Args:
        engine: The active physics engine instance.

    Returns:
        Dict with 'q' and 'v' as plain Python float lists, or empty dict if
        the engine does not implement get_state.
    """
    if not hasattr(engine, "get_state"):
        return {}
    result = engine.get_state()
    if not isinstance(result, tuple | list) or len(result) < 2:
        return {}
    q, v = result[0], result[1]
    return {
        "q": q.tolist() if isinstance(q, np.ndarray) else list(q),
        "v": v.tolist() if isinstance(v, np.ndarray) else list(v),
    }


class SimulationFrame(BaseModel):
    """Single frame of simulation data."""

    time: float
    state: dict[str, Any]
    analysis: dict[str, Any] | None = None


async def _load_simulation_engine(
    engine_manager: object,
    engine_type: str,
    websocket: WebSocket,
) -> object | None:
    """Load and return the physics engine, or send an error and return None.

    Args:
        engine_manager: The engine manager from app state.
        engine_type: Engine type string from the URL path.
        websocket: The active WebSocket connection.

    Returns:
        The active physics engine, or None if loading failed.
    """
    if not (engine_manager is not None):
        raise ValueError("engine_manager must be provided")
    require(
        engine_type is not None and len(engine_type.strip()) > 0,
        "Engine type must be a non-empty string",
        engine_type,
    )
    try:
        enum_type = _engine_type_from_str(engine_type)
        success = engine_manager.switch_engine(enum_type)  # type: ignore[attr-defined]
        if not success:
            raise ValueError("Could not load engine")

        engine = engine_manager.get_active_physics_engine()  # type: ignore[attr-defined]
        if not engine:
            raise ValueError("Could not load engine")

        return engine  # type: ignore[no-any-return]
    except ValueError:
        await websocket.send_json({"error": f"Invalid engine: {engine_type}"})
        return None


async def _handle_client_commands(
    websocket: WebSocket,
) -> str:
    """Check for client commands (stop/pause) with a short timeout.

    Returns:
        One of "continue", "stop", or "pause".
    """
    try:
        msg = await asyncio.wait_for(websocket.receive_json(), timeout=0.001)
        if msg.get("action") == "stop":
            return "stop"
        if msg.get("action") == "pause":
            return "pause"
    except TimeoutError:
        pass  # No message, continue simulation
    return "continue"


async def _wait_for_resume_or_stop(websocket: WebSocket) -> bool:
    """Wait while paused for a resume or stop command.

    Args:
        websocket: The active WebSocket connection.

    Returns:
        True if the simulation should stop, False if it should resume.
    """
    await websocket.send_json({"status": "paused"})
    while True:
        msg = await websocket.receive_json()
        if msg.get("action") == "resume":
            return False
        if msg.get("action") == "stop":
            return True


_TRAJECTORY_SAMPLE_INTERVAL = 50  # Record one state snapshot every N steps


async def _run_simulation_loop(
    websocket: WebSocket,
    engine: object,
    config: dict[str, Any],
) -> tuple[int, float, list[dict[str, Any]], list[float]]:
    """Execute the simulation loop, streaming frames to the client.

    Args:
        websocket: The active WebSocket connection.
        engine: The physics engine instance.
        config: Simulation configuration dict.

    Returns:
        Tuple of (frame_count, time_elapsed, trajectory_samples, peak_torques).
        peak_torques holds the per-joint absolute maximum observed over the run.
    """
    if not (websocket is not None):
        raise ValueError("websocket must be provided")
    duration = config.get("duration", 3.0)
    timestep = config.get("timestep", 0.002)

    require(duration > 0, "Simulation duration must be positive", duration)
    require(timestep > 0, "Simulation timestep must be positive", timestep)

    await websocket.send_json({"status": "running", "duration": duration})

    time_elapsed = 0.0
    frame = 0
    trajectory_samples: list[dict[str, Any]] = []
    peak_torques: list[float] = []

    # Calculate frame skip for ~60fps UI updates
    target_fps = 60
    steps_per_second = 1.0 / timestep
    frame_skip = max(1, int(steps_per_second / target_fps))

    while time_elapsed < duration:
        command = await _handle_client_commands(websocket)
        if command == "stop":
            break
        if command == "pause":
            stopped = await _wait_for_resume_or_stop(websocket)
            if stopped:
                break

        # Step simulation
        if hasattr(engine, "step"):
            engine.step(timestep)

        time_elapsed += timestep
        frame += 1

        # Track peak per-joint torques over the entire run
        if hasattr(engine, "get_torques"):
            raw = engine.get_torques()
            if isinstance(raw, list | np.ndarray):
                arr = np.asarray(raw, dtype=float)
                step_abs = [float(abs(v)) for v in arr]
                if not peak_torques:
                    peak_torques = step_abs
                else:
                    peak_torques = [
                        max(a, b) for a, b in zip(peak_torques, step_abs, strict=False)
                    ]

        # Collect sparse trajectory samples for the RunSummary
        if frame % _TRAJECTORY_SAMPLE_INTERVAL == 0:
            state = _engine_state_to_dict(engine)
            trajectory_samples.append({"time": round(time_elapsed, 4), **state})

        # Send frame data (throttle to ~60fps for UI)
        if frame % frame_skip == 0:
            state = _engine_state_to_dict(engine)

            frame_data: dict[str, Any] = {
                "frame": frame,
                "time": round(time_elapsed, 4),
                "state": state,
            }

            # Include analysis if requested
            if config.get("live_analysis"):
                frame_data["analysis"] = {
                    "joint_angles": (
                        engine.get_joint_angles()
                        if hasattr(engine, "get_joint_angles")
                        else None
                    ),
                    "velocities": (
                        engine.get_velocities()
                        if hasattr(engine, "get_velocities")
                        else None
                    ),
                }

            await websocket.send_json(frame_data)

    return frame, time_elapsed, trajectory_samples, peak_torques


def _build_run_summary(
    engine_type: str,
    duration_s: float,
    steps: int,
    engine: object,
    trajectory: list[dict[str, Any]],
    peak_torques: list[float] | None = None,
) -> dict[str, Any]:
    """Build a RunSummary dict from post-loop engine state.

    Args:
        engine_type: Engine identifier string.
        duration_s: Total simulated time.
        steps: Number of simulation steps completed.
        engine: The physics engine instance.
        trajectory: Sparse trajectory sample list.
        peak_torques: Per-joint absolute peak torques tracked during the run.
            When provided, used directly; otherwise falls back to a single
            post-loop sample from the engine.

    Returns:
        RunSummary-compatible dict ready for JSON serialisation.
    """
    if peak_torques is not None:
        max_torques = peak_torques
    elif hasattr(engine, "get_torques"):
        raw = engine.get_torques()
        if isinstance(raw, list | np.ndarray):
            arr = np.asarray(raw, dtype=float)
            max_torques = [float(abs(v)) for v in arr]
        else:
            max_torques = []
    else:
        max_torques = []

    energy = 0.0
    if hasattr(engine, "get_energy"):
        raw_energy = engine.get_energy()
        if isinstance(raw_energy, int | float):
            energy = float(raw_energy)

    next_steps: list[str] = []
    if duration_s > 0 and steps > 0:
        if max_torques and max(max_torques) > 100.0:
            next_steps.append(
                "High peak torques detected — consider reducing duration or timestep"
            )
        if energy > 1000.0:
            next_steps.append("High mechanical energy — review initial conditions")
        if not next_steps:
            next_steps.append(
                "Run looks nominal — try increasing duration or adjusting parameters"
            )

    return RunSummary(
        engine=engine_type,
        duration_s=round(duration_s, 4),
        steps=steps,
        max_torques=max_torques,
        energy=round(energy, 4),
        trajectory=trajectory,
        next_steps=next_steps,
    ).model_dump()


@router.websocket("/ws/simulate/{engine_type}")
async def simulation_stream(
    websocket: WebSocket,
    engine_type: str,
) -> None:
    """
    Stream simulation in real-time over WebSocket.

    Client sends: {"action": "start", "config": {...}}
    Server sends: {"frame": 0, "time": 0.0, "state": {...}, ...}

    No authentication required in local mode.
    """
    if not (websocket is not None):
        raise ValueError("websocket must be provided")
    await websocket.accept()

    # Access engine manager from app state
    engine_manager = websocket.app.state.engine_manager

    try:
        # Wait for start command
        start_msg = await websocket.receive_json()

        if start_msg.get("action") != "start":
            await websocket.send_json({"error": "Expected 'start' action"})
            return

        config = start_msg.get("config", {})

        # Load engine
        engine = await _load_simulation_engine(engine_manager, engine_type, websocket)
        if engine is None:
            return

        # Set initial state if provided, using the (q, v) engine contract
        if "initial_state" in config:
            _apply_initial_state(engine, config["initial_state"])

        # Run simulation loop
        frame, time_elapsed, trajectory, peak_torques = await _run_simulation_loop(
            websocket, engine, config
        )

        # Build post-simulation summary
        summary = _build_run_summary(
            engine_type=engine_type,
            duration_s=time_elapsed,
            steps=frame,
            engine=engine,
            trajectory=trajectory,
            peak_torques=peak_torques,
        )

        # Send completion with embedded RunSummary
        await websocket.send_json(
            {
                "status": "complete",
                "total_frames": frame,
                "total_time": round(time_elapsed, 4),
                "summary": summary,
            }
        )

    except WebSocketDisconnect:
        pass  # Client disconnected
    except (ValueError, RuntimeError, AttributeError) as e:
        # Best effort error reporting
        with contextlib.suppress(ConnectionError, TimeoutError, OSError):
            await websocket.send_json({"error": str(e)})
    finally:
        with contextlib.suppress(ConnectionError, TimeoutError, OSError):
            await websocket.close()
