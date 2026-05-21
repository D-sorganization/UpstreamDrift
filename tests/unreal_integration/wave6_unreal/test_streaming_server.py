"""Wave6 async tests for UnrealStreamingServer with mocked clients.

The server is started on port 0 (OS-assigned) and exercised end-to-end
without any real Unreal Engine on the other side. Clients are fakes that
record everything they receive.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

import pytest

from src.unreal_integration._streaming_config import (
    ControlAction,
    ControlMessage,
    StreamingConfig,
    StreamingState,
)
from src.unreal_integration._streaming_server import UnrealStreamingServer
from src.unreal_integration.data_frame import UnrealDataFrame
from src.unreal_integration.geometry import Quaternion, Vector3
from src.unreal_integration.skeleton import JointState

pytestmark = pytest.mark.asyncio


class _FakeClient:
    def __init__(self, fail_after: int | None = None) -> None:
        self.received: list[str] = []
        self.fail_after = fail_after
        self._sent_count = 0

    async def send(self, data: str) -> None:
        if self.fail_after is not None and self._sent_count >= self.fail_after:
            raise OSError("simulated disconnect")
        self._sent_count += 1
        self.received.append(data)


def _frame(n: int = 0) -> UnrealDataFrame:
    return UnrealDataFrame(
        timestamp=float(n) * 0.01,
        frame_number=n,
        joints={
            "x": JointState(
                name="x", position=Vector3(), rotation=Quaternion.identity()
            )
        },
    )


@pytest.fixture
def cfg() -> StreamingConfig:
    return StreamingConfig(host="127.0.0.1", port=0, buffer_size=5)


async def test_initial_state_stopped(cfg: StreamingConfig) -> None:
    s = UnrealStreamingServer(config=cfg)
    assert s.state == StreamingState.STOPPED
    assert s.client_count == 0
    assert s.bound_port == 0
    assert s.playback_speed == 1.0


async def test_start_stop_lifecycle(cfg: StreamingConfig) -> None:
    s = UnrealStreamingServer(config=cfg)
    await s.start()
    try:
        assert s.state == StreamingState.RUNNING
        assert s.bound_port > 0
    finally:
        await s.stop()
    assert s.state == StreamingState.STOPPED


async def test_async_context_manager(cfg: StreamingConfig) -> None:
    async with UnrealStreamingServer(config=cfg) as s:
        assert s.state == StreamingState.RUNNING
    assert s.state == StreamingState.STOPPED


async def test_start_twice_raises(cfg: StreamingConfig) -> None:
    s = UnrealStreamingServer(config=cfg)
    await s.start()
    try:
        with pytest.raises(RuntimeError, match="Cannot start"):
            await s.start()
    finally:
        await s.stop()


async def test_start_bind_failure_sets_error_state() -> None:
    # Negative port → OSError → wrapped in RuntimeError
    s = UnrealStreamingServer(
        config=(
            StreamingConfig(host="127.0.0.1", port=70001)
            if False
            else StreamingConfig(host="127.0.0.1", port=0)
        )
    )
    # Force OSError by replacing asyncio.start_server
    import src.unreal_integration._streaming_server as srv_mod

    async def fake_start(*_a: object, **_kw: object) -> None:
        raise OSError("bind failed")

    orig = srv_mod.asyncio.start_server
    srv_mod.asyncio.start_server = fake_start  # type: ignore[assignment]
    try:
        with pytest.raises(RuntimeError, match="Cannot bind"):
            await s.start()
        assert s.state == StreamingState.ERROR
    finally:
        srv_mod.asyncio.start_server = orig  # type: ignore[assignment]


async def test_stop_when_already_stopped_is_noop(cfg: StreamingConfig) -> None:
    s = UnrealStreamingServer(config=cfg)
    await s.stop()  # should not raise
    assert s.state == StreamingState.STOPPED


async def test_broadcast_sends_to_all_clients(cfg: StreamingConfig) -> None:
    s = UnrealStreamingServer(config=cfg)
    await s.start()
    try:
        c1 = _FakeClient()
        c2 = _FakeClient()
        await s._add_client(c1)
        await s._add_client(c2)
        await s.broadcast(_frame(1))
        assert len(c1.received) == 1
        assert len(c2.received) == 1
        msg = json.loads(c1.received[0])
        assert msg["type"] == "frame"
        assert msg["data"]["frame"] == 1
    finally:
        await s.stop()


async def test_broadcast_when_not_running_does_nothing(cfg: StreamingConfig) -> None:
    s = UnrealStreamingServer(config=cfg)
    c1 = _FakeClient()
    s._clients.add(c1)
    await s.broadcast(_frame(0))
    assert c1.received == []


async def test_broadcast_drops_failing_clients(cfg: StreamingConfig) -> None:
    s = UnrealStreamingServer(config=cfg)
    await s.start()
    try:
        good = _FakeClient()
        bad = _FakeClient(fail_after=0)
        await s._add_client(good)
        await s._add_client(bad)
        await s.broadcast(_frame(0))
        # bad should be removed; good still in
        assert s.client_count == 1
        assert good in s._clients
    finally:
        await s.stop()


async def test_max_clients_enforced() -> None:
    s = UnrealStreamingServer(
        config=StreamingConfig(host="127.0.0.1", port=0, max_clients=1)
    )
    await s.start()
    try:
        c1, c2 = _FakeClient(), _FakeClient()
        await s._add_client(c1)
        await s._add_client(c2)
        assert s.client_count == 1
    finally:
        await s.stop()


async def test_callbacks_fire(cfg: StreamingConfig) -> None:
    s = UnrealStreamingServer(config=cfg)
    connect = MagicMock()
    disconnect = MagicMock()
    s.on_client_connect(connect)
    s.on_client_disconnect(disconnect)
    await s.start()
    try:
        c = _FakeClient()
        await s._add_client(c)
        await s._remove_client(c)
        connect.assert_called_once()
        disconnect.assert_called_once()
    finally:
        await s.stop()


async def test_handle_control_pause_play(cfg: StreamingConfig) -> None:
    s = UnrealStreamingServer(config=cfg)
    await s.start()
    try:
        await s._handle_control(ControlMessage(action=ControlAction.PAUSE))
        assert s.state == StreamingState.PAUSED
        await s._handle_control(ControlMessage(action=ControlAction.PLAY))
        assert s.state == StreamingState.RUNNING
    finally:
        await s.stop()


async def test_handle_control_set_speed(cfg: StreamingConfig) -> None:
    s = UnrealStreamingServer(config=cfg)
    cb = MagicMock()
    s.on_control_message(cb)
    await s._handle_control(ControlMessage(action=ControlAction.SET_SPEED, value=2.5))
    assert s.playback_speed == 2.5
    cb.assert_called_once()


async def test_handle_control_seek(cfg: StreamingConfig) -> None:
    s = UnrealStreamingServer(config=cfg)
    await s._handle_control(ControlMessage(action=ControlAction.SEEK, value=4.0))
    assert s._current_time == 4.0


async def test_handle_control_reset(cfg: StreamingConfig) -> None:
    s = UnrealStreamingServer(config=cfg)
    s._buffer.add(_frame(0))
    s._frames_sent = 5
    s._current_time = 3.0
    await s._handle_control(ControlMessage(action=ControlAction.RESET))
    assert s._buffer.is_empty
    assert s._frames_sent == 0
    assert s._current_time == 0.0


async def test_handle_control_stop(cfg: StreamingConfig) -> None:
    s = UnrealStreamingServer(config=cfg)
    await s.start()
    await s._handle_control(ControlMessage(action=ControlAction.STOP))
    assert s.state == StreamingState.STOPPED


async def test_queue_frame_adds_to_buffer(cfg: StreamingConfig) -> None:
    s = UnrealStreamingServer(config=cfg)
    s.queue_frame(_frame(0))
    s.queue_frame(_frame(1))
    assert len(s._buffer) == 2


async def test_get_statistics_keys(cfg: StreamingConfig) -> None:
    s = UnrealStreamingServer(config=cfg)
    stats = s.get_statistics()
    for key in (
        "state",
        "clients_connected",
        "frames_sent",
        "uptime",
        "average_fps",
        "buffer_size",
        "playback_speed",
        "current_time",
    ):
        assert key in stats


async def test_real_tcp_client_connects(cfg: StreamingConfig) -> None:
    """Smoke test: open a real TCP connection on loopback and verify
    the server tracks the client. Confirms the _handle_new_connection
    wiring works."""
    s = UnrealStreamingServer(config=cfg)
    await s.start()
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", s.bound_port)
        # Give the server a moment to register the client
        await asyncio.sleep(0.05)
        assert s.client_count == 1
        writer.close()
        await writer.wait_closed()
        # Allow disconnect bookkeeping
        await asyncio.sleep(0.05)
        assert s.client_count == 0
        # silence unused
        _ = reader
    finally:
        await s.stop()
