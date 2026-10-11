"""WebSocket simulation loop force overlay streaming tests (#11307, FTO-22)."""

from unittest.mock import AsyncMock

import numpy as np
import pytest

from src.api.routes import simulation_ws
from src.shared.python.force_overlay import (
    ForceTorqueFrame,
    OverlayWrench,
    WrenchKind,
)

pytestmark = pytest.mark.unit


class MockStreamingEngine:
    """Mock engine satisfying ForceTorqueProvider."""

    def __init__(self) -> None:
        self.time = 0.0

    def step(self, dt: float) -> None:
        self.time += dt

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(1), np.zeros(1)

    def get_force_torque_frame(self) -> ForceTorqueFrame:
        wrench = OverlayWrench(
            kind=WrenchKind.CONTACT,
            label="contact:ground",
            body="foot",
            point_m=(0.0, 0.0, 0.0),
            force_n=(0.0, 0.0, 750.0),
            source="stream_mock",
        )
        return ForceTorqueFrame(
            time_s=self.time,
            engine="stream_mock",
            wrenches=(wrench,),
        )


class MockWebSocket:
    """Mock WebSocket sink."""

    def __init__(self) -> None:
        self.send_json = AsyncMock()


@pytest.mark.asyncio
async def test_stream_carries_force_overlay_when_opted_in(monkeypatch) -> None:
    socket = MockWebSocket()
    monkeypatch.setattr(
        simulation_ws, "_handle_client_commands", AsyncMock(return_value=None)
    )

    engine = MockStreamingEngine()
    config = {
        "duration": 0.04,
        "timestep": 0.02,
        "force_overlay": True,
        "force_overlay_style": {
            "force_types": ["contact"],
            "scale_factor": 0.02,
        },
    }

    await simulation_ws._run_simulation_loop(socket, engine, config)

    # Inspect the transmitted frames
    assert socket.send_json.call_count >= 2
    last_frame_payload = socket.send_json.call_args_list[-1].args[0]

    assert "force_overlay" in last_frame_payload
    fo = last_frame_payload["force_overlay"]
    assert fo is not None
    assert fo["glyphs"] is not None
    assert fo["frame"] is not None
    assert fo["glyphs"]["time_s"] == last_frame_payload["time"]
    assert len(fo["glyphs"]["arrows"]) == 1
    assert fo["glyphs"]["arrows"][0]["label"] == "contact:ground"


@pytest.mark.asyncio
async def test_stream_omits_force_overlay_when_not_opted_in(monkeypatch) -> None:
    socket = MockWebSocket()
    monkeypatch.setattr(
        simulation_ws, "_handle_client_commands", AsyncMock(return_value=None)
    )

    engine = MockStreamingEngine()
    config = {
        "duration": 0.04,
        "timestep": 0.02,
        # force_overlay not present
    }

    await simulation_ws._run_simulation_loop(socket, engine, config)

    assert socket.send_json.call_count >= 2
    last_frame_payload = socket.send_json.call_args_list[-1].args[0]

    assert "force_overlay" not in last_frame_payload
