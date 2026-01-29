"""WebSocket routes for real-time simulation streaming."""

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from src.shared.python.engine_registry import EngineType

router = APIRouter()


class SimulationFrame(BaseModel):
    """Single frame of simulation data."""

    time: float
    state: dict
    analysis: dict | None = None


@router.websocket("/ws/simulate/{engine_type}")
async def simulation_stream(
    websocket: WebSocket,
    engine_type: str,
):
    """
    Stream simulation in real-time over WebSocket.

    Client sends: {"action": "start", "config": {...}}
    Server sends: {"frame": 0, "time": 0.0, "state": {...}, ...}

    No authentication required in local mode.
    """
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
        try:
            enum_type = EngineType(engine_type.upper())
            # Use public interface if available, else private
            if hasattr(engine_manager, "load_engine"):
                engine_manager.load_engine(enum_type)
            else:
                engine_manager._load_engine(enum_type)

            engine = engine_manager.get_active_physics_engine()
            if not engine:
                raise ValueError("Could not load engine")

        except ValueError:
            await websocket.send_json({"error": f"Invalid engine: {engine_type}"})
            return

        # Set initial state if provided
        if "initial_state" in config and hasattr(engine, "set_state"):
            engine.set_state(config["initial_state"])

        # Simulation parameters
        duration = config.get("duration", 3.0)
        timestep = config.get("timestep", 0.002)

        await websocket.send_json({"status": "running", "duration": duration})

        # Run simulation, streaming frames
        time_elapsed = 0.0
        frame = 0

        while time_elapsed < duration:
            # Check for client commands (pause, stop, etc.)
            try:
                msg = await asyncio.wait_for(websocket.receive_json(), timeout=0.001)
                if msg.get("action") == "stop":
                    break
                if msg.get("action") == "pause":
                    await websocket.send_json({"status": "paused"})
                    # Wait for resume
                    while True:
                        msg = await websocket.receive_json()
                        if msg.get("action") == "resume":
                            break
                        if msg.get("action") == "stop":
                            raise StopIteration
            except TimeoutError:
                pass  # No message, continue simulation

            # Step simulation
            if hasattr(engine, "step"):
                engine.step(timestep)

            time_elapsed += timestep
            frame += 1

            # Send frame data (throttle to 60fps for UI)
            if frame % max(1, int(1 / (60 * timestep))) == 0:
                state = {}
                if hasattr(engine, "get_state"):
                    state = engine.get_state()

                frame_data = {
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

        # Send completion
        await websocket.send_json(
            {
                "status": "complete",
                "total_frames": frame,
                "total_time": round(time_elapsed, 4),
            }
        )

    except WebSocketDisconnect:
        pass  # Client disconnected
    except StopIteration:
        await websocket.send_json({"status": "stopped"})
    except Exception as e:
        # Best effort error reporting
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
