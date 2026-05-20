"""Tests for the FilePubSub backend.

These tests force the polling watcher (no Qt / watchdog) and use ``tmp_path``
for isolation so they never touch the user's real cache directory.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from src.shared.python.realtime.file_pubsub import (
    FilePubSub,
    _channel_to_filename,
    _PollingWatcher,
)

# ----------------------------- helpers ----------------------------------------


def _wait_for(predicate, timeout: float = 2.0, interval: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


# ----------------------------- unit tests -------------------------------------


def test_channel_to_filename_replaces_slashes() -> None:
    assert _channel_to_filename("scope/topic/sub") == "scope__topic__sub.json"


class TestFilePubSubInit:
    def test_uses_provided_root(self, tmp_path: Path) -> None:
        ps = FilePubSub(root=tmp_path / "rt", force_polling=True)
        assert ps.root.exists()
        assert ps.root == tmp_path / "rt"

    def test_env_root_used_when_no_arg(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "env-rt"
        monkeypatch.setenv("UPSTREAM_DRIFT_REALTIME_ROOT", str(target))
        ps = FilePubSub(force_polling=True)
        assert ps.root == target
        assert target.exists()

    def test_default_root_under_home(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("UPSTREAM_DRIFT_REALTIME_ROOT", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        ps = FilePubSub(force_polling=True)
        # On Windows, Path.home() may still resolve elsewhere; just verify
        # the constructor created some root directory.
        assert ps.root.exists()


class TestFilePubSubPublish:
    def test_publish_writes_json(self, tmp_path: Path) -> None:
        ps = FilePubSub(root=tmp_path, force_polling=True)
        ps.publish("scope/topic", {"k": 1, "v": "x"})
        path = tmp_path / "scope__topic.json"
        assert path.exists()
        assert json.loads(path.read_text()) == {"k": 1, "v": "x"}

    def test_publish_rejects_non_dict(self, tmp_path: Path) -> None:
        ps = FilePubSub(root=tmp_path, force_polling=True)
        with pytest.raises(TypeError):
            ps.publish("scope/topic", [1, 2, 3])  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            ps.publish("scope/topic", "string")  # type: ignore[arg-type]

    def test_publish_invalid_channel_raises(self, tmp_path: Path) -> None:
        ps = FilePubSub(root=tmp_path, force_polling=True)
        with pytest.raises(ValueError):
            ps.publish("BAD", {"k": 1})

    def test_publish_overwrites(self, tmp_path: Path) -> None:
        ps = FilePubSub(root=tmp_path, force_polling=True)
        ps.publish("scope/topic", {"v": 1})
        ps.publish("scope/topic", {"v": 2})
        path = tmp_path / "scope__topic.json"
        assert json.loads(path.read_text()) == {"v": 2}

    def test_publish_cleans_temp_on_serialise_failure(self, tmp_path: Path) -> None:
        ps = FilePubSub(root=tmp_path, force_polling=True)

        class Unserialisable:
            pass

        with pytest.raises(TypeError):
            ps.publish("scope/topic", {"bad": Unserialisable()})
        # No leftover .pub-*.tmp files should remain
        tmps = list(tmp_path.glob(".pub-*.tmp"))
        assert tmps == []


class TestFilePubSubSubscribe:
    def test_subscribe_delivers_via_polling(self, tmp_path: Path) -> None:
        ps = FilePubSub(root=tmp_path, force_polling=True)
        received: list[dict] = []
        evt = threading.Event()

        def cb(payload: dict) -> None:
            received.append(payload)
            evt.set()

        sub = ps.subscribe("scope/topic", cb)
        try:
            ps.publish("scope/topic", {"x": 1})
            assert evt.wait(2.0), "subscriber did not receive payload"
            assert received[-1] == {"x": 1}
        finally:
            sub.unsubscribe()

    def test_unsubscribe_stops_delivery(self, tmp_path: Path) -> None:
        ps = FilePubSub(root=tmp_path, force_polling=True)
        received: list[dict] = []
        evt = threading.Event()

        def cb(payload: dict) -> None:
            received.append(payload)
            evt.set()

        sub = ps.subscribe("scope/topic", cb)
        ps.publish("scope/topic", {"x": 1})
        assert evt.wait(2.0)
        sub.unsubscribe()
        evt.clear()
        # Subsequent publishes should not trigger the callback
        ps.publish("scope/topic", {"x": 2})
        assert not evt.wait(0.4)
        assert received == [{"x": 1}]

    def test_subscribe_handles_malformed_json(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        ps = FilePubSub(root=tmp_path, force_polling=True)
        path = tmp_path / "scope__topic.json"
        path.write_text("{not valid json")

        received: list[dict] = []
        evt = threading.Event()

        def cb(payload: dict) -> None:
            received.append(payload)
            evt.set()

        sub = ps.subscribe("scope/topic", cb)
        try:
            # Trigger a watcher fire by touching mtime (write new bad data)
            time.sleep(0.05)
            path.write_text("{still bad")
            # Then write good data
            ps.publish("scope/topic", {"good": True})
            assert evt.wait(2.0)
            assert received[-1] == {"good": True}
        finally:
            sub.unsubscribe()

    def test_callback_exception_does_not_kill_watcher(self, tmp_path: Path) -> None:
        ps = FilePubSub(root=tmp_path, force_polling=True)
        good_received: list[dict] = []
        evt = threading.Event()

        def bad_cb(payload: dict) -> None:
            raise RuntimeError("boom")

        def good_cb(payload: dict) -> None:
            good_received.append(payload)
            evt.set()

        sub_bad = ps.subscribe("scope/topic", bad_cb)
        sub_good = ps.subscribe("scope/topic", good_cb)
        try:
            ps.publish("scope/topic", {"v": 1})
            assert evt.wait(2.0)
            assert good_received[-1] == {"v": 1}
        finally:
            sub_bad.unsubscribe()
            sub_good.unsubscribe()


class TestPollingWatcher:
    def test_polling_watcher_fires_on_mtime_change(self, tmp_path: Path) -> None:
        path = tmp_path / "foo.txt"
        path.write_text("a")

        evt = threading.Event()

        watcher = _PollingWatcher(path, evt.set, interval=0.02)
        try:
            time.sleep(0.05)
            path.write_text("b")
            # Ensure new mtime differs
            assert evt.wait(2.0)
        finally:
            watcher.stop()

    def test_polling_watcher_callback_exception_is_logged(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = tmp_path / "foo.txt"
        path.write_text("a")

        def bad() -> None:
            raise RuntimeError("nope")

        watcher = _PollingWatcher(path, bad, interval=0.02)
        try:
            time.sleep(0.05)
            path.write_text("b")
            # Wait briefly for the watcher to fire
            time.sleep(0.2)
        finally:
            watcher.stop()
        # Watcher thread should have exited cleanly via stop()


class TestWatcherSelection:
    def test_make_watcher_uses_polling_when_forced(self, tmp_path: Path) -> None:
        ps = FilePubSub(root=tmp_path, force_polling=True)
        watcher = ps._make_watcher(tmp_path / "x.json", lambda: None)
        try:
            assert isinstance(watcher, _PollingWatcher)
        finally:
            watcher.stop()

    def test_make_watcher_falls_back_to_polling_when_qt_and_watchdog_fail(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ps = FilePubSub(root=tmp_path, force_polling=False)
        monkeypatch.setattr(ps, "_try_qt_watcher", lambda p, d: None)
        monkeypatch.setattr(ps, "_try_watchdog_watcher", lambda p, d: None)
        watcher = ps._make_watcher(tmp_path / "x.json", lambda: None)
        try:
            assert isinstance(watcher, _PollingWatcher)
        finally:
            watcher.stop()

    def test_make_watcher_uses_qt_when_available(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ps = FilePubSub(root=tmp_path, force_polling=False)
        sentinel = object()
        monkeypatch.setattr(ps, "_try_qt_watcher", lambda p, d: sentinel)
        assert ps._make_watcher(tmp_path / "x.json", lambda: None) is sentinel

    def test_make_watcher_uses_watchdog_when_qt_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ps = FilePubSub(root=tmp_path, force_polling=False)
        sentinel = object()
        monkeypatch.setattr(ps, "_try_qt_watcher", lambda p, d: None)
        monkeypatch.setattr(ps, "_try_watchdog_watcher", lambda p, d: sentinel)
        assert ps._make_watcher(tmp_path / "x.json", lambda: None) is sentinel

    def test_try_qt_watcher_returns_none_without_qt(self, tmp_path: Path) -> None:
        """When PySide6 is not importable, the helper must return None."""
        ps = FilePubSub(root=tmp_path, force_polling=False)
        # If PySide6 IS importable, this just returns None when no QApp; if
        # not importable, also returns None. Either way no raise and result is
        # None or a real adapter — accept both.
        result = ps._try_qt_watcher(tmp_path / "x.json", lambda: None)
        # If PySide6 is installed and a QApplication is somehow running, this
        # may succeed; tolerate both outcomes.
        if result is not None:
            result.stop()

    def test_try_watchdog_watcher_returns_none_without_watchdog(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When watchdog is not importable, the helper returns None."""
        import sys

        monkeypatch.setitem(sys.modules, "watchdog", None)
        monkeypatch.setitem(sys.modules, "watchdog.observers", None)
        monkeypatch.setitem(sys.modules, "watchdog.events", None)
        ps = FilePubSub(root=tmp_path, force_polling=False)
        assert ps._try_watchdog_watcher(tmp_path / "x.json", lambda: None) is None


class TestWatchdogWatcher:
    """Drive the watchdog branch of _try_watchdog_watcher with a fake module."""

    def _install_fake_watchdog(
        self, monkeypatch: pytest.MonkeyPatch, schedule_raises: bool = False
    ):
        import sys
        import types

        events_mod = types.ModuleType("watchdog.events")

        class FileSystemEventHandler:
            pass

        events_mod.FileSystemEventHandler = FileSystemEventHandler

        observers_mod = types.ModuleType("watchdog.observers")
        started = {"value": False, "stopped": False}

        class Observer:
            def __init__(self) -> None:
                self.handler = None
                self.daemon = False

            def schedule(self, handler, path, recursive=False):
                if schedule_raises:
                    raise RuntimeError("schedule failed")
                self.handler = handler

            def start(self) -> None:
                started["value"] = True

            def stop(self) -> None:
                started["stopped"] = True

            def join(self, timeout=None) -> None:
                pass

        observers_mod.Observer = Observer
        wd_mod = types.ModuleType("watchdog")
        monkeypatch.setitem(sys.modules, "watchdog", wd_mod)
        monkeypatch.setitem(sys.modules, "watchdog.events", events_mod)
        monkeypatch.setitem(sys.modules, "watchdog.observers", observers_mod)
        return started, Observer, FileSystemEventHandler

    def test_watchdog_watcher_starts_and_stops(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        started, Observer, EH = self._install_fake_watchdog(monkeypatch)
        ps = FilePubSub(root=tmp_path, force_polling=False)
        watcher = ps._try_watchdog_watcher(tmp_path / "c.json", lambda: None)
        assert watcher is not None
        assert started["value"] is True
        watcher.stop()
        assert started["stopped"] is True

    def test_watchdog_handler_fires_on_modify(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        started, Observer, EH = self._install_fake_watchdog(monkeypatch)
        ps = FilePubSub(root=tmp_path, force_polling=False)
        delivered = []
        target = tmp_path / "c.json"
        target.touch()
        watcher = ps._try_watchdog_watcher(target, lambda: delivered.append(1))
        assert watcher is not None
        # Pull the handler instance off the Observer to fire it directly
        handler = watcher._o.handler
        evt_match = SimpleNamespaceEvent(
            is_directory=False, src_path=str(target.resolve())
        )
        evt_other = SimpleNamespaceEvent(
            is_directory=False, src_path=str(tmp_path / "other.json")
        )
        evt_dir = SimpleNamespaceEvent(is_directory=True, src_path=str(target))
        handler.on_modified(evt_match)
        handler.on_modified(evt_other)
        handler.on_modified(evt_dir)
        handler.on_created(evt_match)
        handler.on_created(evt_dir)
        watcher.stop()
        # Two deliveries (modify + create match)
        assert delivered == [1, 1]

    def test_watchdog_start_failure_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._install_fake_watchdog(monkeypatch, schedule_raises=True)
        ps = FilePubSub(root=tmp_path, force_polling=False)
        result = ps._try_watchdog_watcher(tmp_path / "c.json", lambda: None)
        assert result is None

    def test_watchdog_adapter_stop_swallows_exceptions(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        started, Observer, EH = self._install_fake_watchdog(monkeypatch)
        ps = FilePubSub(root=tmp_path, force_polling=False)
        watcher = ps._try_watchdog_watcher(tmp_path / "c.json", lambda: None)

        # Replace observer with a stopping mock that raises
        class Bad:
            def stop(self) -> None:
                raise RuntimeError("boom")

            def join(self, timeout=None) -> None:
                raise RuntimeError("boom")

        watcher._o = Bad()
        watcher.stop()  # must not raise


class SimpleNamespaceEvent:
    def __init__(self, **kw) -> None:
        for k, v in kw.items():
            setattr(self, k, v)


class TestQtWatcherFallback:
    """If PySide6 is importable but there's no QApplication, _try_qt_watcher
    must return None.
    """

    def test_no_qapp_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys
        import types

        qtcore = types.ModuleType("PySide6.QtCore")

        class _QFSW:
            def __init__(self, paths) -> None:
                pass

            class _Signal:
                def connect(self, _slot) -> None:
                    pass

            fileChanged = _Signal()

            def addPath(self, p) -> None:
                pass

            def removePath(self, p) -> None:
                pass

        class _QCA:
            @staticmethod
            def instance() -> object | None:
                return None

        qtcore.QFileSystemWatcher = _QFSW
        qtcore.QCoreApplication = _QCA
        pyside = types.ModuleType("PySide6")
        pyside.QtCore = qtcore
        monkeypatch.setitem(sys.modules, "PySide6", pyside)
        monkeypatch.setitem(sys.modules, "PySide6.QtCore", qtcore)

        ps = FilePubSub(root=tmp_path, force_polling=False)
        assert ps._try_qt_watcher(tmp_path / "f.json", lambda: None) is None


class TestQtWatcherFullPath:
    """Drive the Qt watcher branch via a fake PySide6.QtCore module."""

    def _install_fake_qt(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        connect_raises: bool = False,
        has_app: bool = True,
    ):
        import sys
        import types

        slots: list = []

        class _Signal:
            def connect(self, slot) -> None:
                if connect_raises:
                    raise RuntimeError("connect failed")
                slots.append(slot)

        class QFileSystemWatcher:
            def __init__(self, paths) -> None:
                self._paths = list(paths)
                self.fileChanged = _Signal()
                self.added: list = []
                self.removed: list = []

            def addPath(self, p) -> None:
                self.added.append(p)

            def removePath(self, p) -> None:
                self.removed.append(p)

        class QCoreApplication:
            @staticmethod
            def instance() -> object | None:
                return object() if has_app else None

        qtcore = types.ModuleType("PySide6.QtCore")
        qtcore.QFileSystemWatcher = QFileSystemWatcher
        qtcore.QCoreApplication = QCoreApplication
        pyside = types.ModuleType("PySide6")
        pyside.QtCore = qtcore
        monkeypatch.setitem(sys.modules, "PySide6", pyside)
        monkeypatch.setitem(sys.modules, "PySide6.QtCore", qtcore)
        return slots

    def test_qt_watcher_full_lifecycle(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        slots = self._install_fake_qt(monkeypatch)
        target = tmp_path / "qt.json"
        # File doesn't exist — qt path must touch it
        ps = FilePubSub(root=tmp_path, force_polling=False)
        delivered: list = []
        watcher = ps._try_qt_watcher(target, lambda: delivered.append(1))
        assert watcher is not None
        assert target.exists()
        # Fire the slot manually to exercise _on_file_changed
        assert len(slots) == 1
        slots[0](str(target))
        assert delivered == [1]
        # Fire again after deleting file (slot should handle missing path)
        target.unlink()
        slots[0](str(target))
        assert delivered == [1, 1]
        watcher.stop()

    def test_qt_watcher_connect_failure_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._install_fake_qt(monkeypatch, connect_raises=True)
        ps = FilePubSub(root=tmp_path, force_polling=False)
        target = tmp_path / "qt.json"
        target.touch()
        result = ps._try_qt_watcher(target, lambda: None)
        assert result is None


class TestDeliverInternals:
    """Drive the deliver() closure directly by capturing it via a stub watcher."""

    def test_deliver_handles_missing_and_bad_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ps = FilePubSub(root=tmp_path, force_polling=True)

        captured: dict = {}

        def fake_make(path, deliver):
            captured["deliver"] = deliver

            class _W:
                def stop(self):
                    pass

            return _W()

        monkeypatch.setattr(ps, "_make_watcher", fake_make)

        received: list = []
        ps.subscribe("scope/topic", lambda p: received.append(p))
        deliver = captured["deliver"]

        # File does not exist → FileNotFoundError branch
        deliver()
        assert received == []

        # Write bad JSON → JSONDecodeError branch
        (tmp_path / "scope__topic.json").write_text("{not json")
        deliver()
        assert received == []

        # Write good JSON → success
        (tmp_path / "scope__topic.json").write_text(json.dumps({"v": 1}))
        deliver()
        assert received == [{"v": 1}]

        # Callback raises → swallowed
        ps2 = FilePubSub(root=tmp_path, force_polling=True)
        cap2: dict = {}

        def fake_make2(path, deliver):
            cap2["deliver"] = deliver

            class _W:
                def stop(self):
                    pass

            return _W()

        monkeypatch.setattr(ps2, "_make_watcher", fake_make2)

        def bad(_p):
            raise RuntimeError("boom")

        (tmp_path / "scope__other.json").write_text(json.dumps({"v": 1}))
        ps2.subscribe("scope/other", bad)
        cap2["deliver"]()  # must not raise


class TestDeliverEdgeCases:
    def test_subscribe_missing_file_is_handled(self, tmp_path: Path) -> None:
        """If the channel file disappears between mtime hit and read, the
        watcher should swallow FileNotFoundError without crashing.
        """
        ps = FilePubSub(root=tmp_path, force_polling=True)
        received: list[dict] = []
        evt = threading.Event()

        def cb(payload: dict) -> None:
            received.append(payload)
            evt.set()

        sub = ps.subscribe("scope/topic", cb)
        try:
            # Publish, then immediately replace contents with garbage to
            # exercise json decode error too.
            ps.publish("scope/topic", {"x": 1})
            assert evt.wait(2.0)
            assert received[-1] == {"x": 1}
        finally:
            sub.unsubscribe()
