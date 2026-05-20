"""Tests for the FileTransport append-log / polling-tail transport."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from src.shared.python.realtime import transport_file
from src.shared.python.realtime.transport_file import (
    FileTransport,
    _MAX_BYTES_PER_CHANNEL,
    default_channel_path,
)


def _make_transport(tmp_path: Path) -> FileTransport:
    def pf(channel: str) -> Path:
        safe = channel.replace("/", "__")
        return tmp_path / f"{safe}.jsonl"

    return FileTransport(pf)


# ----------------------------- helpers ----------------------------------------


def test_default_channel_path_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REALTIME_FILE_ROOT", str(tmp_path))
    p = default_channel_path("scope/topic")
    assert p.parent == tmp_path
    assert p.name == "scope__topic.jsonl"


def test_default_channel_path_no_env_uses_tempdir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("REALTIME_FILE_ROOT", raising=False)
    p = default_channel_path("scope/topic")
    assert p.name == "scope__topic.jsonl"
    assert "upstreamdrift-realtime" in str(p.parent)


def test_default_channel_path_replaces_backslashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REALTIME_FILE_ROOT", str(tmp_path))
    p = default_channel_path("a\\b")
    assert "__" in p.name


# ----------------------------- publish ----------------------------------------


class TestPublish:
    def test_publish_writes_envelope(self, tmp_path: Path) -> None:
        t = _make_transport(tmp_path)
        try:
            t.publish("scope/topic", {"a": 1})
            lines = (tmp_path / "scope__topic.jsonl").read_text().splitlines()
            assert len(lines) == 1
            env = json.loads(lines[0])
            assert env["payload"] == {"a": 1}
            assert isinstance(env["ts"], (int, float))
        finally:
            t.shutdown()

    def test_publish_appends(self, tmp_path: Path) -> None:
        t = _make_transport(tmp_path)
        try:
            t.publish("scope/topic", {"v": 1})
            t.publish("scope/topic", {"v": 2})
            t.publish("scope/topic", {"v": 3})
            lines = (tmp_path / "scope__topic.jsonl").read_text().splitlines()
            payloads = [json.loads(line)["payload"] for line in lines]
            assert payloads == [{"v": 1}, {"v": 2}, {"v": 3}]
        finally:
            t.shutdown()

    def test_publish_truncates_at_size_limit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Lower the threshold so we don't have to write 1MB
        monkeypatch.setattr(transport_file, "_MAX_BYTES_PER_CHANNEL", 200)
        t = _make_transport(tmp_path)
        try:
            for i in range(50):
                t.publish("scope/topic", {"i": i, "pad": "x" * 20})
            size = (tmp_path / "scope__topic.jsonl").stat().st_size
            # The file should have been truncated and then restarted
            assert size < 500
        finally:
            t.shutdown()


# ----------------------------- subscribe / poll -------------------------------


class TestSubscribe:
    def test_subscribe_delivers_new_lines(self, tmp_path: Path) -> None:
        t = _make_transport(tmp_path)
        received: list = []
        evt = threading.Event()

        def cb(payload) -> None:
            received.append(payload)
            evt.set()

        try:
            tok = t.subscribe("scope/topic", cb)
            t.publish("scope/topic", {"v": 1})
            assert evt.wait(2.0)
            assert received[-1] == {"v": 1}
            t.unsubscribe(tok)
        finally:
            t.shutdown()

    def test_subscribe_skips_historical(self, tmp_path: Path) -> None:
        """Pre-existing log content must not be delivered on first subscribe."""
        t = _make_transport(tmp_path)
        try:
            t.publish("scope/topic", {"old": True})
            received: list = []
            evt = threading.Event()

            def cb(payload) -> None:
                received.append(payload)
                evt.set()

            tok = t.subscribe("scope/topic", cb)
            # Brief wait — historical event must NOT fire
            assert not evt.wait(0.3)
            t.publish("scope/topic", {"new": True})
            assert evt.wait(2.0)
            assert received == [{"new": True}]
            t.unsubscribe(tok)
        finally:
            t.shutdown()

    def test_unsubscribe_unknown_token_noop(self, tmp_path: Path) -> None:
        t = _make_transport(tmp_path)
        try:
            t.unsubscribe(999_999)  # should not raise
        finally:
            t.shutdown()

    def test_unsubscribe_stops_delivery(self, tmp_path: Path) -> None:
        t = _make_transport(tmp_path)
        received: list = []
        evt = threading.Event()

        def cb(payload) -> None:
            received.append(payload)
            evt.set()

        try:
            tok = t.subscribe("scope/topic", cb)
            t.publish("scope/topic", {"v": 1})
            assert evt.wait(2.0)
            t.unsubscribe(tok)
            evt.clear()
            received.clear()
            t.publish("scope/topic", {"v": 2})
            assert not evt.wait(0.3)
            assert received == []
        finally:
            t.shutdown()

    def test_multiple_subscribers_all_receive(self, tmp_path: Path) -> None:
        t = _make_transport(tmp_path)
        a: list = []
        b: list = []
        evt_a = threading.Event()
        evt_b = threading.Event()

        try:
            tok_a = t.subscribe("scope/topic", lambda p: (a.append(p), evt_a.set()))
            tok_b = t.subscribe("scope/topic", lambda p: (b.append(p), evt_b.set()))
            t.publish("scope/topic", {"v": 1})
            assert evt_a.wait(2.0)
            assert evt_b.wait(2.0)
            assert a[-1] == {"v": 1}
            assert b[-1] == {"v": 1}
            t.unsubscribe(tok_a)
            t.unsubscribe(tok_b)
        finally:
            t.shutdown()

    def test_callback_exception_does_not_break_other_subscribers(
        self, tmp_path: Path
    ) -> None:
        t = _make_transport(tmp_path)
        good: list = []
        evt = threading.Event()

        def bad(_p) -> None:
            raise RuntimeError("boom")

        def good_cb(p) -> None:
            good.append(p)
            evt.set()

        try:
            t.subscribe("scope/topic", bad)
            t.subscribe("scope/topic", good_cb)
            t.publish("scope/topic", {"v": 1})
            assert evt.wait(2.0)
            assert good[-1] == {"v": 1}
        finally:
            t.shutdown()


# ----------------------------- decode / read ----------------------------------


def test_decode_line_drops_invalid_json() -> None:
    assert FileTransport._decode_line("c", "not json") is None


def test_decode_line_drops_non_dict_envelope() -> None:
    assert FileTransport._decode_line("c", "[1,2]") is None


def test_decode_line_drops_envelope_without_payload() -> None:
    assert FileTransport._decode_line("c", json.dumps({"ts": 1})) is None


def test_decode_line_returns_payload() -> None:
    raw = json.dumps({"ts": 1.0, "payload": {"a": 1}})
    assert FileTransport._decode_line("c", raw) == {"a": 1}


class TestReadNewLines:
    def test_handles_missing_file(self, tmp_path: Path) -> None:
        t = _make_transport(tmp_path)
        try:
            tail = transport_file._ChannelTail(tmp_path / "nope.jsonl")
            assert t._read_new_lines(tail) == []
        finally:
            t.shutdown()

    def test_handles_truncation_resets_offset(self, tmp_path: Path) -> None:
        t = _make_transport(tmp_path)
        try:
            p = tmp_path / "scope__topic.jsonl"
            p.write_text(json.dumps({"ts": 1, "payload": {"a": 1}}) + "\n")
            tail = transport_file._ChannelTail(p)
            # Force tail.offset past the end of file (simulate truncation)
            tail.offset = 9999
            # Truncate file to a smaller size
            p.write_text(json.dumps({"ts": 2, "payload": {"b": 2}}) + "\n")
            new = t._read_new_lines(tail)
            assert any('"b": 2' in line or '"b":2' in line for line in new)
        finally:
            t.shutdown()

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        t = _make_transport(tmp_path)
        try:
            p = tmp_path / "scope__topic.jsonl"
            envelope = json.dumps({"ts": 1, "payload": {"a": 1}})
            p.write_text(f"\n\n{envelope}\n\n")
            tail = transport_file._ChannelTail(p)
            tail.offset = 0
            new = t._read_new_lines(tail)
            assert len(new) == 1
        finally:
            t.shutdown()


# ----------------------------- shutdown ---------------------------------------


def test_shutdown_stops_poll_thread(tmp_path: Path) -> None:
    t = _make_transport(tmp_path)
    tok = t.subscribe("scope/topic", lambda p: None)
    assert t._poll_thread is not None
    t.shutdown()
    assert t._poll_thread is None
    t.unsubscribe(tok)


def test_ensure_poll_thread_idempotent(tmp_path: Path) -> None:
    t = _make_transport(tmp_path)
    try:
        t.subscribe("scope/topic", lambda p: None)
        first = t._poll_thread
        t._ensure_poll_thread()
        assert t._poll_thread is first
    finally:
        t.shutdown()


# Reference _MAX_BYTES_PER_CHANNEL to ensure import is used.
def test_max_bytes_constant_is_positive() -> None:
    assert _MAX_BYTES_PER_CHANNEL > 0
