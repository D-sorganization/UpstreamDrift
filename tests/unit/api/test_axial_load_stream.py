"""Actual WebSocket loop carries optional signed segment frames through JSON."""

from unittest.mock import AsyncMock

import numpy as np
import pytest

from src.api.routes import simulation_ws
from src.shared.python.body_part_viz.axial_loads import AxialLoadFrame


@pytest.mark.asyncio
async def test_stream_preserves_force_frame_and_wire_time(monkeypatch):
    class Engine:
        time = 0.0

        def step(self, dt):
            self.time += dt

        def get_state(self):
            return np.zeros(1), np.zeros(1)

        def get_segment_axial_loads(self):
            return AxialLoadFrame(
                self.time, {"custom-link": -25.0}, "fixture proximal reaction"
            )

    class Socket:
        send_json = AsyncMock()

    socket = Socket()
    monkeypatch.setattr(
        simulation_ws, "_handle_client_commands", AsyncMock(return_value=None)
    )
    await simulation_ws._run_simulation_loop(
        socket, Engine(), {"duration": 0.020123, "timestep": 0.020123}
    )
    payload = socket.send_json.call_args_list[-1].args[0]
    assert payload["segment_loads"]["values_n"] == {"custom-link": -25.0}
    assert payload["segment_loads"]["time_s"] == payload["time"]
    assert payload["segment_loads"]["units"] == "N"
