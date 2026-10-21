"""File-based realtime transport.

Canonical home: the fleet's provider tree, Tools
``src/shared/python/realtime/`` — UpstreamDrift consumes this module
through its ``vendor/ud-tools`` pin (UpstreamDrift#8942).

Each channel maps to a JSON-line append-log under a per-user cache
directory. Publishers append through one persistent per-channel handle;
subscribers tail with a tracked byte offset. Tailing runs on a single
shared daemon thread that polls every ``_POLL_INTERVAL_S`` seconds and
fans out to per-channel callbacks.

Hot-path contract (UpstreamDrift#8942 Defect A): directory setup and
handle opening run once per channel — not once per message — and the
subscriber reads only bytes past its offset, never re-parsing delivered
content.

The transport is intentionally simple — it's a hint layer, not a
broker. The file transport is the right choice for the launcher demo
because it works across processes started in either order and across
crashes; a websocket transport would be a follow-up if we ever need
sub-30 ms latency.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any

from ..logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

__all__ = ["FileTransport", "default_channel_path"]


_POLL_INTERVAL_S = 1.0 / 30.0  # 30 Hz tail polling
_MAX_BYTES_PER_CHANNEL = 1_000_000  # truncate beyond ~1 MB to avoid runaway logs


def _root_dir() -> Path:
    """Return the per-user realtime root directory.

    Honors ``REALTIME_FILE_ROOT`` (set by tests for isolation), else
    falls back to the OS temp dir under ``upstreamdrift-realtime``.
    """
    override = os.environ.get("REALTIME_FILE_ROOT")
    if override:
        return Path(override)
    return Path(tempfile.gettempdir()) / "upstreamdrift-realtime"


def default_channel_path(channel: str) -> Path:
    """Return the path of the JSON-line log for *channel*."""
    safe = channel.replace("/", "__").replace("\\", "__")
    return _root_dir() / f"{safe}.jsonl"


class _ChannelTail:
    """Per-channel tail state."""

    __slots__ = ("path", "offset", "callbacks")

    def __init__(self, path: Path) -> None:
        self.path = path
        # Start from the current end-of-file so we only deliver future
        # messages (not the entire history).
        try:
            self.offset = path.stat().st_size if path.exists() else 0
        except OSError:
            self.offset = 0
        self.callbacks: dict[int, Callable[[Any], None]] = {}


class FileTransport:
    """Append-log + polling-tail realtime transport."""

    def __init__(
        self,
        path_for_channel: Callable[[str], Path],
        *,
        poll_thread: bool = True,
    ) -> None:
        self._path_for_channel = path_for_channel
        self._poll_thread_enabled = poll_thread
        self._lock = threading.RLock()
        self._channels: dict[str, _ChannelTail] = {}
        self._next_token = 1
        self._token_to_channel: dict[int, str] = {}
        self._poll_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        # One append handle per channel; opened lazily on first publish
        # (or after a truncation rollover) and never per message.
        self._append_handles: dict[str, IO[str]] = {}
        # Bytes appended since the last size-bound check, per channel.
        self._appended_since_check: dict[str, int] = {}

    # ---- publish ------------------------------------------------------

    def publish(self, channel: str, payload: Any) -> None:
        """Serialise *payload* and append it to *channel*'s log."""
        line = json.dumps(
            {"ts": time.time(), "payload": payload}, separators=(",", ":")
        )
        with self._lock:
            handle = self._append_handles.get(channel)
            if handle is None or handle.closed:
                handle = self._open_append(channel)
            handle.write(line)
            handle.write("\n")
            handle.flush()
            appended = self._appended_since_check.get(channel, 0) + len(line) + 1
            self._appended_since_check[channel] = appended
            if appended >= _MAX_BYTES_PER_CHANNEL:
                self._truncate_if_oversized(channel)

    def _open_append(self, channel: str) -> IO[str]:
        """Open the per-channel append handle, creating parents once.

        mkdir/open run only on the first publish for a channel (or after
        a truncation rollover closed the handle) — never per message.
        """
        path = self._path_for_channel(channel)
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a", encoding="utf-8")
        self._append_handles[channel] = handle
        self._appended_since_check[channel] = 0
        return handle

    def _truncate_if_oversized(self, channel: str) -> None:
        """Bound a runaway log to :data:`_MAX_BYTES_PER_CHANNEL`.

        Runs only when roughly ``_MAX_BYTES_PER_CHANNEL`` new bytes have
        been appended since the last check, so the exists/stat cost stays
        off the per-message hot path. Overshoot is bounded by one message.
        """
        self._appended_since_check[channel] = 0
        path = self._path_for_channel(channel)
        try:
            if not path.exists() or path.stat().st_size <= _MAX_BYTES_PER_CHANNEL:
                return
        except OSError:
            return
        handle = self._append_handles.pop(channel, None)
        if handle is not None:
            handle.close()
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning(
                "realtime: could not truncate oversized channel log %s", path
            )
            return
        tail = self._channels.get(channel)
        if tail is not None:
            tail.offset = 0

    # ---- subscribe ----------------------------------------------------

    def subscribe(self, channel: str, callback: Callable[[Any], None]) -> int:
        """Register *callback* on *channel* and return an unsubscribe token."""
        with self._lock:
            tail = self._channels.get(channel)
            if tail is None:
                tail = _ChannelTail(self._path_for_channel(channel))
                self._channels[channel] = tail
            token = self._next_token
            self._next_token += 1
            tail.callbacks[token] = callback
            self._token_to_channel[token] = channel
            self._ensure_poll_thread()
        return token

    def unsubscribe(self, token: int) -> None:
        should_stop = False
        with self._lock:
            channel = self._token_to_channel.pop(token, None)
            if channel is None:
                return
            tail = self._channels.get(channel)
            if tail is None:
                return
            tail.callbacks.pop(token, None)
            if not tail.callbacks:
                self._channels.pop(channel, None)
            if not self._channels:
                should_stop = True
        if should_stop:
            self._stop_and_join_poll_thread()

    # ---- polling ------------------------------------------------------

    def _ensure_poll_thread(self) -> None:
        if not self._poll_thread_enabled:
            return
        if self._poll_thread is not None and self._poll_thread.is_alive():
            return
        self._stop_event.clear()
        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            name="realtime-file-tail",
            daemon=True,
        )
        self._poll_thread.start()

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._poll_once()
            except Exception:  # pragma: no cover - defensive
                logger.exception("realtime tail poll iteration failed")
            self._stop_event.wait(_POLL_INTERVAL_S)

    def _poll_once(self) -> None:
        # Snapshot under lock to keep iteration safe; dispatch outside
        # the lock so callbacks cannot deadlock against (un)subscribe.
        with self._lock:
            tails = list(self._channels.items())
        for channel, tail in tails:
            new_lines = self._read_new_lines(tail)
            if not new_lines:
                continue
            with self._lock:
                # Re-snapshot callbacks; some may have unsubscribed
                # between the file read and dispatch.
                callbacks = list(tail.callbacks.values())
            for raw in new_lines:
                payload = self._decode_line(channel, raw)
                if payload is None:
                    continue
                for cb in callbacks:
                    try:
                        cb(payload)
                    except Exception:  # pragma: no cover - callback isolation
                        logger.exception(
                            "realtime callback raised on channel %s", channel
                        )

    def _read_new_lines(self, tail: _ChannelTail) -> list[str]:
        """Read and commit bytes past ``tail.offset`` (offset-tracked).

        The offset read-modify-write is serialized under the transport
        lock so concurrent polls (thread + explicit ``_poll_once``) can
        never both claim the same bytes and double-deliver.
        """
        with self._lock:
            try:
                stat = tail.path.stat()
            except FileNotFoundError:
                return []
            except OSError:
                return []
            if stat.st_size < tail.offset:
                # Log was truncated/rotated under us; reset.
                tail.offset = 0
            if stat.st_size == tail.offset:
                return []
            try:
                with tail.path.open("r", encoding="utf-8") as fh:
                    fh.seek(tail.offset)
                    chunk = fh.read()
                    tail.offset = fh.tell()
            except OSError:
                return []
        if not chunk:
            return []
        return [line for line in chunk.splitlines() if line.strip()]

    @staticmethod
    def _decode_line(channel: str, raw: str) -> Any | None:
        try:
            envelope = json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("realtime: dropping unparseable line on channel %s", channel)
            return None
        if not isinstance(envelope, dict) or "payload" not in envelope:
            return None
        return envelope["payload"]

    def _stop_and_join_poll_thread(self) -> None:
        """Stop and join the poll thread if running and caller is not it."""
        thread_to_join: threading.Thread | None = None
        with self._lock:
            self._stop_event.set()
            if (
                self._poll_thread is not None
                and threading.current_thread() != self._poll_thread
            ):
                thread_to_join = self._poll_thread
            self._poll_thread = None
        if thread_to_join is not None:
            thread_to_join.join(timeout=1.0)

    def shutdown(self) -> None:
        """Stop the poll thread and release per-channel append handles."""
        self._stop_and_join_poll_thread()
        for handle in self._append_handles.values():
            try:
                handle.close()
            except OSError:  # pragma: no cover - already-closed handle
                pass
        self._append_handles.clear()
