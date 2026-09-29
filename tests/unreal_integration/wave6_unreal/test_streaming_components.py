"""Wave6 fast tests for streaming buffer, config, and protocol.

These exercise the in-process pieces of the streaming stack — no real
sockets, no event loop required. The async server itself is exercised
elsewhere; here we focus on pure-Python helpers.
"""

from __future__ import annotations

import json

import pytest

from src.unreal_integration._streaming_buffer import FrameBuffer
from src.unreal_integration._streaming_config import (
    ControlAction,
    ControlMessage,
    StreamingConfig,
    StreamingState,
)
from src.unreal_integration._streaming_protocol import StreamingProtocol
from src.unreal_integration.data_frame import UnrealDataFrame
from src.unreal_integration.geometry import Quaternion, Vector3
from src.unreal_integration.skeleton import JointState


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


# ---------- StreamingState ----------


class TestStreamingState:
    def test_is_active_running(self) -> None:
        assert StreamingState.RUNNING.is_active is True

    def test_is_active_paused(self) -> None:
        assert StreamingState.PAUSED.is_active is True

    def test_is_active_stopped(self) -> None:
        assert StreamingState.STOPPED.is_active is False

    def test_is_active_error(self) -> None:
        assert StreamingState.ERROR.is_active is False


# ---------- ControlAction / ControlMessage ----------


class TestControlAction:
    def test_from_string_known(self) -> None:
        assert ControlAction.from_string("pause") == ControlAction.PAUSE

    def test_from_string_case_insensitive(self) -> None:
        assert ControlAction.from_string("PLAY") == ControlAction.PLAY

    def test_from_string_unknown(self) -> None:
        with pytest.raises(ValueError, match="Unknown control action"):
            ControlAction.from_string("frobnicate")


class TestControlMessage:
    def test_to_json_basic(self) -> None:
        msg = ControlMessage(action=ControlAction.PAUSE)
        d = json.loads(msg.to_json())
        assert d["type"] == "control"
        assert d["action"] == "pause"
        assert "value" not in d

    def test_to_json_with_value_and_client(self) -> None:
        msg = ControlMessage(action=ControlAction.SEEK, value=2.5, client_id="abc")
        d = json.loads(msg.to_json())
        assert d["value"] == 2.5
        assert d["client_id"] == "abc"

    def test_from_json_roundtrip(self) -> None:
        msg = ControlMessage(action=ControlAction.SET_SPEED, value=2.0)
        msg2 = ControlMessage.from_json(msg.to_json())
        assert msg2.action == ControlAction.SET_SPEED
        assert msg2.value == 2.0


# ---------- StreamingConfig ----------


class TestStreamingConfig:
    def test_defaults(self) -> None:
        c = StreamingConfig()
        assert c.host == "localhost"
        assert c.port == 8765
        assert c.frame_interval == pytest.approx(1 / 60)

    def test_invalid_port(self) -> None:
        with pytest.raises(ValueError, match="port"):
            StreamingConfig(port=70000)
        with pytest.raises(ValueError, match="port"):
            StreamingConfig(port=-1)

    def test_invalid_fps(self) -> None:
        with pytest.raises(ValueError, match="fps"):
            StreamingConfig(target_fps=0)

    def test_invalid_buffer(self) -> None:
        with pytest.raises(ValueError, match="buffer"):
            StreamingConfig(buffer_size=0)

    def test_to_from_dict_roundtrip(self) -> None:
        c = StreamingConfig(
            host="0.0.0.0",
            port=9000,
            target_fps=30,
            buffer_size=20,
            enable_compression=True,
            heartbeat_interval=2.0,
            max_clients=5,
            enable_metrics=False,
        )
        c2 = StreamingConfig.from_dict(c.to_dict())
        assert c2.host == "0.0.0.0"
        assert c2.port == 9000
        assert c2.enable_compression is True
        assert c2.enable_metrics is False


# ---------- FrameBuffer ----------


class TestFrameBuffer:
    def test_empty(self) -> None:
        b = FrameBuffer(max_size=3)
        assert b.is_empty
        assert not b.is_full
        assert len(b) == 0
        assert b.get() is None
        assert b.peek() is None

    def test_add_and_get(self) -> None:
        b = FrameBuffer(max_size=3)
        f0 = _frame(0)
        b.add(f0)
        assert len(b) == 1
        assert b.peek() is f0
        # peek must not remove
        assert len(b) == 1
        assert b.get() is f0
        assert b.is_empty

    def test_overflow_drops_oldest(self) -> None:
        b = FrameBuffer(max_size=2)
        f0, f1, f2 = _frame(0), _frame(1), _frame(2)
        b.add(f0)
        b.add(f1)
        assert b.is_full
        b.add(f2)
        # f0 should have been dropped
        all_frames = b.get_all()
        assert all_frames[0] is f1
        assert all_frames[1] is f2

    def test_clear(self) -> None:
        b = FrameBuffer(max_size=2)
        b.add(_frame(0))
        b.clear()
        assert b.is_empty

    def test_get_all_does_not_remove(self) -> None:
        b = FrameBuffer(max_size=4)
        b.add(_frame(0))
        b.add(_frame(1))
        all_frames = b.get_all()
        assert len(all_frames) == 2
        assert len(b) == 2

    def test_invalid_max_size(self) -> None:
        # Regression: previously accepted bogus max_size of 0
        with pytest.raises(ValueError, match="max_size"):
            FrameBuffer(max_size=0)


# ---------- StreamingProtocol ----------


class TestStreamingProtocol:
    def test_frame_message(self) -> None:
        msg = StreamingProtocol.create_frame_message(_frame(7))
        assert msg["type"] == "frame"
        assert msg["data"]["frame"] == 7

    def test_status_message(self) -> None:
        msg = StreamingProtocol.create_status_message(
            state=StreamingState.RUNNING, fps=30.0, frames_sent=10, buffer_size=4
        )
        assert msg["type"] == "status"
        assert msg["state"] == "running"
        assert msg["frames_sent"] == 10
        assert msg["buffer_size"] == 4

    def test_error_message_with_details(self) -> None:
        msg = StreamingProtocol.create_error_message(
            "E_BAD", "something failed", details={"key": "value"}
        )
        assert msg["type"] == "error"
        assert msg["error_code"] == "E_BAD"
        assert msg["details"] == {"key": "value"}

    def test_error_message_without_details(self) -> None:
        msg = StreamingProtocol.create_error_message("E_X", "msg")
        assert "details" not in msg

    def test_ack_message(self) -> None:
        msg = StreamingProtocol.create_ack_message(frame_number=10, timestamp=1.5)
        assert msg == {"type": "ack", "frame_number": 10, "timestamp": 1.5}

    def test_heartbeat_message(self) -> None:
        msg = StreamingProtocol.create_heartbeat_message()
        assert msg["type"] == "heartbeat"
        assert isinstance(msg["server_time"], float)
