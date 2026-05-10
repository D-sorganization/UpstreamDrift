"""File-based publish/subscribe backend.

Each channel maps to a single JSON file under the storage root (default
``~/.upstream_drift/realtime/``). Publishing performs an atomic
write-then-rename. Subscribers are notified via:

1. ``QFileSystemWatcher`` if Qt is importable, else
2. ``watchdog.observers.Observer`` if available, else
3. a 100 ms polling thread.

Latency budget: < 200 ms one-hop on a warm cache.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .protocol import Subscription, validate_channel

__all__ = ["FilePubSub"]


logger = logging.getLogger(__name__)


def _channel_to_filename(channel: str) -> str:
    """Encode ``scope/topic/sub`` as ``scope__topic__sub.json``."""
    return channel.replace("/", "__") + ".json"


class _PollingWatcher:
    """Threaded fallback watcher that polls a file's mtime."""

    def __init__(
        self,
        path: Path,
        on_change: Callable[[], None],
        interval: float = 0.1,
    ) -> None:
        self._path = path
        self._on_change = on_change
        self._interval = interval
        self._stop = threading.Event()
        self._last_mtime: float = -1.0
        try:
            self._last_mtime = path.stat().st_mtime
        except OSError:
            self._last_mtime = -1.0
        self._thread = threading.Thread(
            target=self._run,
            name=f"realtime-poll-{path.name}",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                mtime = self._path.stat().st_mtime
            except OSError:
                mtime = -1.0
            if mtime != self._last_mtime and mtime > 0:
                self._last_mtime = mtime
                try:
                    self._on_change()
                except Exception:
                    logger.exception("polling watcher callback failed")
            self._stop.wait(self._interval)

    def stop(self) -> None:
        self._stop.set()
        # Avoid joining ourselves if we somehow ended up on the watcher thread.
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=1.0)


class FilePubSub:
    """File-based pub-sub backend.

    Args:
        root: Storage directory; defaults to ``$UPSTREAM_DRIFT_REALTIME_ROOT``
            or ``~/.upstream_drift/realtime``.
        force_polling: If True, always use the polling fallback even when Qt or
            watchdog is available. Useful for unit tests.
    """

    def __init__(
        self,
        root: Path | str | None = None,
        *,
        force_polling: bool = False,
    ) -> None:
        if root is None:
            env_root = os.environ.get("UPSTREAM_DRIFT_REALTIME_ROOT")
            if env_root:
                root = Path(env_root)
            else:
                root = Path.home() / ".upstream_drift" / "realtime"
        self.root: Path = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._force_polling = force_polling
        # Track active watchers per file path so multiple subscribers on the
        # same channel can share infrastructure where appropriate.
        self._lock = threading.RLock()

    # -- file path helpers ---------------------------------------------------

    def _path_for(self, channel: str) -> Path:
        validate_channel(channel)
        return self.root / _channel_to_filename(channel)

    # -- publish -------------------------------------------------------------

    def publish(self, channel: str, payload: dict) -> None:
        """Atomically write ``payload`` as JSON to the channel's file."""
        if not isinstance(payload, dict):
            raise TypeError(f"payload must be a dict, got {type(payload).__name__}")
        path = self._path_for(channel)
        # Atomic write: temp file in same directory, then os.replace.
        fd, tmp_name = tempfile.mkstemp(
            prefix=".pub-",
            suffix=".tmp",
            dir=str(self.root),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f)
                f.flush()
                # fsync not available on every fs; non-fatal.
                with contextlib.suppress(OSError):
                    os.fsync(f.fileno())
            os.replace(tmp_name, path)
        except Exception:
            # Clean up temp file if rename failed.
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
            raise

    # -- subscribe -----------------------------------------------------------

    def subscribe(
        self,
        channel: str,
        callback: Callable[[dict], None],
    ) -> Subscription:
        """Subscribe ``callback`` to writes on ``channel``'s file."""
        path = self._path_for(channel)

        def deliver() -> None:
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
            except FileNotFoundError:
                return
            except json.JSONDecodeError:
                # Mid-write race; skip this notification.
                return
            try:
                callback(data)
            except Exception:
                logger.exception(
                    "file pub-sub subscriber callback raised for %s", channel
                )

        watcher = self._make_watcher(path, deliver)

        def _unsubscribe() -> None:
            with self._lock:
                watcher.stop()

        return Subscription(
            channel=channel, callback=callback, _unsubscribe=_unsubscribe
        )

    # -- watcher selection ---------------------------------------------------

    def _make_watcher(self, path: Path, deliver: Callable[[], None]) -> Any:
        if self._force_polling:
            return _PollingWatcher(path, deliver)

        qt_watcher = self._try_qt_watcher(path, deliver)
        if qt_watcher is not None:
            return qt_watcher

        watchdog_watcher = self._try_watchdog_watcher(path, deliver)
        if watchdog_watcher is not None:
            return watchdog_watcher

        return _PollingWatcher(path, deliver)

    def _try_qt_watcher(self, path: Path, deliver: Callable[[], None]) -> Any | None:
        try:  # pragma: no cover - depends on Qt availability
            from PySide6.QtCore import QFileSystemWatcher  # type: ignore

            # QFileSystemWatcher requires a QApplication / QCoreApplication.
            from PySide6.QtCore import QCoreApplication  # type: ignore

            if QCoreApplication.instance() is None:
                return None
        except Exception:
            return None

        # Ensure the file exists so QFileSystemWatcher can observe it.
        if not path.exists():
            try:
                path.touch()
            except OSError:
                return None

        try:
            qfsw = QFileSystemWatcher([str(path)])

            def _on_file_changed(_p: str) -> None:
                # QFileSystemWatcher stops tracking after file is replaced
                # (e.g., via os.replace). Re-arm the watcher after each event.
                qfsw.addPath(_p)
                deliver()

            qfsw.fileChanged.connect(_on_file_changed)
        except Exception:
            return None

        class _QtAdapter:
            def __init__(self, w: Any) -> None:
                self._w = w

            def stop(self) -> None:
                with contextlib.suppress(Exception):
                    self._w.removePath(str(path))

        return _QtAdapter(qfsw)

    def _try_watchdog_watcher(
        self, path: Path, deliver: Callable[[], None]
    ) -> Any | None:
        try:  # pragma: no cover - depends on watchdog availability
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
        except Exception:
            return None

        target = str(path.resolve())

        class _Handler(FileSystemEventHandler):  # type: ignore[misc]
            def on_modified(self, event: Any) -> None:
                if not event.is_directory and os.path.abspath(event.src_path) == target:
                    deliver()

            def on_created(self, event: Any) -> None:
                if not event.is_directory and os.path.abspath(event.src_path) == target:
                    deliver()

        try:
            obs = Observer()
            obs.schedule(_Handler(), str(self.root), recursive=False)
            obs.daemon = True
            obs.start()
        except Exception:
            return None

        class _WatchdogAdapter:
            def __init__(self, o: Any) -> None:
                self._o = o

            def stop(self) -> None:
                try:
                    self._o.stop()
                    self._o.join(timeout=1.0)
                except Exception:
                    pass

        return _WatchdogAdapter(obs)


# Touch ``time`` import to avoid lint error if unused above (used by tests
# importing this module to measure latency).
_ = time
