"""Unit tests for realtime.file_pubsub."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from src.shared.python.realtime.file_pubsub import (
    FilePubSub,
    _channel_to_filename,
)


pytestmark = pytest.mark.unit


def test_channel_to_filename_encoding() -> None:
    assert _channel_to_filename("pose/canonical") == "pose__canonical.json"
    assert _channel_to_filename("engine/mujoco/state") == "engine__mujoco__state.json"


def test_publish_writes_atomic_file(tmp_path: Path) -> None:
    fps = FilePubSub(root=tmp_path, force_polling=True)

    fps.publish("pose/canonical", {"frame": 1, "joints": [0.1, 0.2]})

    expected = tmp_path / "pose__canonical.json"
    assert expected.exists()
    data = json.loads(expected.read_text(encoding="utf-8"))
    assert data == {"frame": 1, "joints": [0.1, 0.2]}


def test_publish_rejects_non_dict(tmp_path: Path) -> None:
    fps = FilePubSub(root=tmp_path, force_polling=True)
    with pytest.raises(TypeError):
        fps.publish("pose/canonical", [1, 2, 3])  # type: ignore[arg-type]


def test_publish_rejects_invalid_channel(tmp_path: Path) -> None:
    fps = FilePubSub(root=tmp_path, force_polling=True)
    with pytest.raises(ValueError):
        fps.publish("BAD/Name", {})


def test_subscribe_round_trip_via_polling(tmp_path: Path) -> None:
    fps = FilePubSub(root=tmp_path, force_polling=True)
    received: list[dict] = []
    event = threading.Event()

    def cb(payload: dict) -> None:
        received.append(payload)
        event.set()

    sub = fps.subscribe("pose/canonical", cb)
    try:
        # Give the polling thread one tick to record initial mtime.
        time.sleep(0.05)
        fps.publish("pose/canonical", {"hello": "world"})
        # Latency budget: < 200 ms; allow generous slack for CI.
        assert event.wait(timeout=2.0), "polling subscriber did not fire"
        assert received == [{"hello": "world"}]
    finally:
        sub.unsubscribe()


def test_multiple_subscribers_all_fire(tmp_path: Path) -> None:
    fps = FilePubSub(root=tmp_path, force_polling=True)
    flags = [threading.Event(), threading.Event(), threading.Event()]
    seen: list[list[dict]] = [[], [], []]

    def make_cb(idx: int):
        def cb(payload: dict) -> None:
            seen[idx].append(payload)
            flags[idx].set()

        return cb

    subs = [fps.subscribe("target/active", make_cb(i)) for i in range(3)]
    try:
        time.sleep(0.05)
        fps.publish("target/active", {"x": 42})
        for f in flags:
            assert f.wait(timeout=2.0)
        for s in seen:
            assert s == [{"x": 42}]
    finally:
        for sub in subs:
            sub.unsubscribe()


def test_unsubscribe_stops_callbacks(tmp_path: Path) -> None:
    fps = FilePubSub(root=tmp_path, force_polling=True)
    counter = {"n": 0}
    fired = threading.Event()

    def cb(payload: dict) -> None:
        counter["n"] += 1
        fired.set()

    sub = fps.subscribe("pose/canonical", cb)
    time.sleep(0.05)
    fps.publish("pose/canonical", {"v": 1})
    assert fired.wait(timeout=2.0)

    sub.unsubscribe()
    before = counter["n"]
    fps.publish("pose/canonical", {"v": 2})
    # Give the (now-stopped) watcher time to *not* fire.
    time.sleep(0.3)
    assert counter["n"] == before


def test_root_auto_created(tmp_path: Path) -> None:
    nested = tmp_path / "nested" / "realtime"
    assert not nested.exists()
    FilePubSub(root=nested, force_polling=True)
    assert nested.is_dir()


def test_env_var_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("UPSTREAM_DRIFT_REALTIME_ROOT", str(tmp_path / "from_env"))
    fps = FilePubSub(force_polling=True)
    assert fps.root == tmp_path / "from_env"
    fps.publish("pose/canonical", {"v": 1})
    assert (tmp_path / "from_env" / "pose__canonical.json").exists()
