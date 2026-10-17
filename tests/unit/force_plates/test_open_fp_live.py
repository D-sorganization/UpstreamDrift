"""OF-09 (#13088): the live open-fp Socket.IO bridge to realtime IPC (fake socket)."""

from __future__ import annotations

import copy
import importlib
import math
import sys
from typing import Any

import pytest

from src.shared.python.force_plates.open_fp_format import live
from src.shared.python.force_plates.open_fp_format.live import (
    LIVE_CHANNEL,
    LiveBridge,
    backoff_delays,
    default_socket_factory,
    register_live_channel,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


class FakeSocket:
    """Records registrations; has no ``emit`` so any call would fail."""

    def __init__(self) -> None:
        self.handlers: dict[str, Any] = {}
        self.connected_to: list[str] = []
        self.disconnects = 0
        self.fail_connects = 0

    def connect(self, url: str) -> None:
        if self.fail_connects:
            self.fail_connects -= 1
            raise ConnectionError("down")
        self.connected_to.append(url)

    def on(self, event: str, handler: Any) -> None:
        self.handlers[event] = handler

    def disconnect(self) -> None:
        self.disconnects += 1


def _batch(**over: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "t": [1.0, 1.005, 1.01],
        "fz": {"L": [100.0, 50.0, 10.0], "R": [200.0, 30.0, 0.0]},
        "cop": {
            "L": [[10.0, -20.0], [11.5, -21.0], None],
            "R": [[-5.0, 0.0], [-6.0, 1.0], None],
            "C": [[0.0, 0.0], [0.0, 0.0], None],
        },
        "lead_pct": [33.33, 62.5, None],
        "forces": [[1.0] * 8, [2.0] * 8, [0.0] * 8],
        "flags": 0,
        "state": "idle",
        "bw_n": 800.0,
        "lead_plate": "L",
        "golfer": "rh",
    }
    body.update(over)
    return body


def _bridge(max_queue: int = 4):
    sock = FakeSocket()
    published: list[tuple[str, dict]] = []
    bridge = LiveBridge(
        "http://192.168.1.5:8000",
        lambda ch, msg: published.append((ch, msg)),
        max_queue=max_queue,
        socket_factory=lambda: sock,
    )
    assert bridge.start() is True
    return bridge, sock, published


def _push(sock: FakeSocket, payload: Any) -> None:
    sock.handlers["live"](payload)


def test_valid_batch_is_published_in_si_units() -> None:
    bridge, sock, published = _bridge()
    _push(sock, _batch())
    assert published == []  # nothing until pumped
    assert bridge.pump() == 1
    channel, msg = published[0]
    assert channel == "force_plate/open_fp/live" == LIVE_CHANNEL
    assert msg["t_s"] == [1.0, 1.005, 1.01]
    assert msg["load_n"] == {"left": [100.0, 50.0, 10.0], "right": [200.0, 30.0, 0.0]}
    assert msg["cop_m"]["left"][0] == [0.01, -0.02]
    assert msg["cop_m"]["right"][1] == [-0.006, 0.001]
    assert msg["cop_m"]["left"][2] is None and msg["cop_m"]["right"][2] is None
    assert msg["dropped"] == 0
    assert msg["source"] == "open-fp-live"
    assert msg["qualified"] is False
    assert bridge.malformed == 0


def test_connect_uses_validated_origin() -> None:
    _, sock, _ = _bridge()
    assert sock.connected_to == ["http://192.168.1.5:8000"]
    assert "live" in sock.handlers


def _mutations() -> dict[str, Any]:
    ragged = _batch()
    ragged["fz"]["L"] = [1.0, 2.0]
    inf_load = _batch()
    inf_load["fz"]["R"][0] = math.inf
    neg_load = _batch()
    neg_load["fz"]["L"][0] = -5.0
    cop_low = _batch()
    cop_low["cop"]["L"][2] = [1.0, 1.0]  # 10 N < 20 N but CoP present
    cop_below_slack = _batch()
    cop_below_slack["fz"]["L"][2] = 19.9
    cop_below_slack["cop"]["L"][2] = [1.0, 1.0]
    cop_missing = _batch()
    cop_missing["cop"]["L"][0] = None  # 100 N but CoP missing
    cop_missing_edge = _batch()
    cop_missing_edge["fz"]["L"][2] = 20.1
    cop_nan = _batch()
    cop_nan["cop"]["L"][0] = [math.nan, 1.0]
    extra_plate = _batch()
    extra_plate["fz"]["X"] = [0.0, 0.0, 0.0]
    return {
        "unknown_key": _batch(surprise=1),
        "ragged": ragged,
        "nan_time": _batch(t=[1.0, math.nan, 1.01]),
        "inf_load": inf_load,
        "decreasing_time": _batch(t=[1.0, 1.01, 1.005]),
        "repeated_time": _batch(t=[1.0, 1.0, 1.01]),
        "negative_load": neg_load,
        "cop_below_threshold": cop_low,
        "cop_below_slack": cop_below_slack,
        "cop_missing_above": cop_missing,
        "cop_missing_above_slack": cop_missing_edge,
        "cop_half_nan": cop_nan,
        "extra_plate": extra_plate,
        "not_a_dict": [1, 2, 3],
        "missing_key": {k: v for k, v in _batch().items() if k != "fz"},
        "string_time": _batch(t=["a", "b", "c"]),
        "bool_time": _batch(t=[True, 2.0, 3.0]),
        "short_forces": _batch(forces=[[1.0] * 8]),
    }


@pytest.mark.parametrize("case", sorted(_mutations()))
def test_malformed_batch_is_dropped_counted_and_never_forwarded(
    case: str, caplog: pytest.LogCaptureFixture
) -> None:
    bridge, sock, published = _bridge()
    with caplog.at_level("WARNING"):
        _push(sock, _mutations()[case])
    assert bridge.malformed == 1
    assert bridge.pump() == 0
    assert published == []
    assert bridge.dropped == 0
    assert caplog.records, "a reason must be logged"


def test_cop_threshold_accepts_rounded_boundary_both_sides() -> None:
    bridge, sock, published = _bridge()
    body = _batch()
    body["fz"]["L"][2] = 19.95  # true load may be 20.0 N rounded: CoP allowed
    body["cop"]["L"][2] = [1.0, 1.0]
    body["fz"]["R"][2] = 20.0  # true load may be 19.96 N rounded: CoP absent
    _push(sock, body)
    bridge.pump()
    assert bridge.malformed == 0 and len(published) == 1


def test_empty_batch_is_ignored_not_malformed() -> None:
    bridge, sock, published = _bridge()
    empty = _batch(
        t=[],
        fz={"L": [], "R": []},
        cop={"L": [], "R": [], "C": []},
        lead_pct=[],
        forces=[],
    )
    _push(sock, empty)
    assert bridge.pump() == 0 and bridge.malformed == 0 and published == []


def test_overflow_drops_oldest_and_reports_drops() -> None:
    bridge, sock, published = _bridge(max_queue=2)
    for k in range(4):
        _push(sock, _batch(t=[k + 1.0, k + 1.005, k + 1.01]))
    assert bridge.dropped == 2
    assert bridge.pump() == 2
    assert [m["t_s"][0] for _, m in published] == [3.0, 4.0]
    assert all(m["dropped"] == 2 for _, m in published)


def test_bridge_never_emits_to_the_board() -> None:
    bridge, sock, _ = _bridge()
    _push(sock, _batch())
    bridge.pump()
    bridge.stop()
    assert not hasattr(sock, "emit")
    assert sock.disconnects == 1


def test_backoff_delays_geometric_and_capped() -> None:
    delays = backoff_delays(base_s=0.5, factor=2.0, cap_s=30.0, attempts=10)
    assert delays[:4] == [0.5, 1.0, 2.0, 4.0]
    assert max(delays) == 30.0 and delays[-1] == 30.0
    assert all(b >= a for a, b in zip(delays, delays[1:], strict=False))
    assert backoff_delays(base_s=0.5, factor=2.0, cap_s=30.0, attempts=0) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"base_s": 0.0},
        {"base_s": math.nan},
        {"factor": 1.0},
        {"cap_s": 0.1},
        {"attempts": -1},
        {"attempts": True},
    ],
)
def test_backoff_delays_rejects_bad_arguments(kwargs: dict[str, Any]) -> None:
    args: dict[str, Any] = {"base_s": 0.5, "factor": 2.0, "cap_s": 30.0, "attempts": 3}
    args.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        backoff_delays(**args)


def _retry_bridge(sock: FakeSocket, sleeps: list[float]) -> LiveBridge:
    return LiveBridge(
        "http://10.0.0.2:8000",
        lambda ch, msg: None,
        max_queue=2,
        socket_factory=lambda: sock,
        sleep=sleeps.append,
    )


def test_start_sleeps_backoff_then_succeeds() -> None:
    sock = FakeSocket()
    sock.fail_connects = 3
    sleeps: list[float] = []
    assert _retry_bridge(sock, sleeps).start(max_attempts=5) is True
    assert sleeps == [0.5, 1.0, 2.0]


def test_start_gives_up_after_max_attempts() -> None:
    sock = FakeSocket()
    sock.fail_connects = 99
    sleeps: list[float] = []
    assert _retry_bridge(sock, sleeps).start(max_attempts=3) is False
    assert sleeps == [0.5, 1.0]


def test_disconnect_event_then_reconnect() -> None:
    sock = FakeSocket()
    bridge = _retry_bridge(sock, [])
    bridge.start()
    assert bridge.connected is True
    sock.handlers["disconnect"]()
    assert bridge.connected is False
    assert bridge.reconnect() is True
    assert bridge.connected is True


@pytest.mark.parametrize(
    "url", ["ftp://h", "http://u:p@h", "http://h/x", "http://h?q=1", ""]
)
def test_bad_base_url_rejected(url: str) -> None:
    with pytest.raises(ValueError):
        LiveBridge(url, lambda c, m: None, max_queue=1)


def test_constructor_validates_types() -> None:
    with pytest.raises(TypeError):
        LiveBridge(5, lambda c, m: None, max_queue=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        LiveBridge("http://h", "nope", max_queue=1)  # type: ignore[arg-type]
    for bad in (0, -1):
        with pytest.raises(ValueError):
            LiveBridge("http://h", lambda c, m: None, max_queue=bad)
    for bad in (True, 1.5):
        with pytest.raises(TypeError):
            LiveBridge("http://h", lambda c, m: None, max_queue=bad)  # type: ignore[arg-type]


def test_channel_registered_in_both_registries() -> None:
    from src.shared.python.realtime import api, channels

    register_live_channel()
    register_live_channel()  # idempotent
    info = api.CHANNEL_REGISTRY[LIVE_CHANNEL]
    assert info.owner_tool_id == "ground_reaction_plots"
    assert "open-fp" in info.description
    assert channels.get_channel_transport(LIVE_CHANNEL) == "ws"


def test_module_imports_without_socketio(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "socketio", None)  # makes import raise
    reloaded = importlib.reload(live)
    assert reloaded.LiveBridge is not None
    with pytest.raises(ImportError, match="force-plate-live"):
        reloaded.default_socket_factory()
    monkeypatch.undo()
    importlib.reload(live)


def test_default_factory_error_names_extra_when_absent() -> None:
    try:
        import socketio  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="force-plate-live"):
            default_socket_factory()
    else:
        pytest.skip("socketio installed")


def test_batch_not_mutated_by_validation() -> None:
    bridge, sock, _ = _bridge()
    body = _batch()
    snapshot = copy.deepcopy(body)
    _push(sock, body)
    assert body == snapshot
