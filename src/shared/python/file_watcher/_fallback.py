"""Pure-Python fallback file watcher built on ``watchdog``.

Mirrors the API of the Rust extension so callers do not need to special-case
which backend is in use. Provides:

- Debouncing (rapid bursts coalesce into a single callback).
- ``.gitignore`` filtering via :mod:`pathspec` if available; otherwise we honor
  a small built-in skip list (``.git``, ``__pycache__``, ``node_modules``,
  ``target``, ``.venv``).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

EventCallback = Callable[[list["ChangeEvent"]], None]

_DEFAULT_SKIP_DIRS = frozenset(
    {".git", "__pycache__", "node_modules", "target", ".venv", ".tox", "dist", "build"}
)


@dataclass(frozen=True)
class ChangeEvent:
    """A single coalesced filesystem event."""

    path: str
    kind: str  # "create" | "modify" | "delete" | "rename"


class FileWatcher:
    """Debounced, gitignore-aware file watcher.

    Parameters
    ----------
    root:
        Project root to watch (recursively).
    debounce_ms:
        Quiet period in milliseconds before flushing accumulated events.
    respect_gitignore:
        If True, drop events whose path matches the project's ``.gitignore``
        (or one of the built-in skip directories).

    Raises
    ------
    ValueError
        If ``root`` does not exist or ``debounce_ms < 0``.
    """

    def __init__(
        self,
        root: str | Path,
        debounce_ms: int = 100,
        respect_gitignore: bool = True,
    ) -> None:
        root_path = Path(root)
        if not root_path.exists():
            raise ValueError(f"root path does not exist: {root}")
        if debounce_ms < 0:
            raise ValueError(f"debounce_ms must be >= 0, got {debounce_ms}")

        self._root = root_path.resolve()
        self._debounce = debounce_ms / 1000.0
        self._respect_gitignore = respect_gitignore
        self._callback: EventCallback | None = None
        self._observer: Any = None  # watchdog Observer; created on start()
        self._handler: Any = None
        self._pending: dict[tuple[str, str], ChangeEvent] = {}
        self._lock = threading.Lock()
        self._last_event_at: float | None = None
        self._stop_flag = threading.Event()
        self._flush_thread: threading.Thread | None = None
        self._gitignore_matcher = self._build_gitignore() if respect_gitignore else None

    # ── Public API ──────────────────────────────────────────────────────────

    @property
    def root(self) -> str:
        return str(self._root)

    @property
    def is_running(self) -> bool:
        return self._observer is not None

    def on_change(self, callback: EventCallback) -> EventCallback:
        """Register a callback. Usable as a decorator."""
        self._callback = callback
        return callback

    def start(self) -> None:
        """Start watching. Raises RuntimeError if already running."""
        if self._observer is not None:
            raise RuntimeError("watcher already started")

        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "watchdog is required for the Python fallback file watcher; "
                "install it with `pip install watchdog`, or build the Rust "
                "extension via `cd rust_core/file_watcher && maturin develop`"
            ) from exc

        watcher = self  # bound for the inner class

        class _Handler(FileSystemEventHandler):
            def on_created(self, event) -> None:
                watcher._enqueue(event.src_path, "create")

            def on_modified(self, event) -> None:
                if event.is_directory:
                    return
                watcher._enqueue(event.src_path, "modify")

            def on_deleted(self, event) -> None:
                watcher._enqueue(event.src_path, "delete")

            def on_moved(self, event) -> None:
                watcher._enqueue(event.src_path, "rename")
                if getattr(event, "dest_path", None):
                    watcher._enqueue(event.dest_path, "rename")

        self._handler = _Handler()
        self._observer = Observer()
        self._observer.schedule(self._handler, str(self._root), recursive=True)
        self._observer.start()

        self._stop_flag.clear()
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()

    def stop(self) -> None:
        """Stop watching. Raises RuntimeError if not started."""
        if self._observer is None:
            raise RuntimeError("watcher not started")
        self._stop_flag.set()
        try:
            self._observer.stop()
            self._observer.join(timeout=2.0)
        finally:
            self._observer = None
            self._handler = None
        if self._flush_thread is not None:
            self._flush_thread.join(timeout=2.0)
            self._flush_thread = None
        # Final flush of anything left in the buffer.
        self._flush_now()

    def __enter__(self) -> FileWatcher:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        if self.is_running:
            self.stop()

    # ── Internals ──────────────────────────────────────────────────────────

    def _enqueue(self, raw_path: str, kind: str) -> None:
        path = str(Path(raw_path))
        if self._should_ignore(path):
            return
        with self._lock:
            self._pending[(path, kind)] = ChangeEvent(path=path, kind=kind)
            self._last_event_at = time.monotonic()

    def _flush_loop(self) -> None:
        poll = max(self._debounce / 4.0, 0.005)
        while not self._stop_flag.is_set():
            time.sleep(poll)
            self._maybe_flush()

    def _maybe_flush(self) -> None:
        with self._lock:
            if not self._pending or self._last_event_at is None:
                return
            if time.monotonic() - self._last_event_at < self._debounce:
                return
            batch = list(self._pending.values())
            self._pending.clear()
            self._last_event_at = None
        self._dispatch(batch)

    def _flush_now(self) -> None:
        with self._lock:
            if not self._pending:
                return
            batch = list(self._pending.values())
            self._pending.clear()
            self._last_event_at = None
        self._dispatch(batch)

    def _dispatch(self, batch: list[ChangeEvent]) -> None:
        if self._callback is None:
            return
        try:
            self._callback(batch)
        except Exception:
            logger.exception("file_watcher callback raised")

    def _should_ignore(self, path: str) -> bool:
        try:
            rel = Path(path).resolve().relative_to(self._root)
        except ValueError:
            return False
        parts = rel.parts
        if any(p in _DEFAULT_SKIP_DIRS for p in parts):
            return True
        if self._gitignore_matcher is not None:
            return bool(self._gitignore_matcher(rel.as_posix()))
        return False

    def _build_gitignore(self) -> Callable[[str], bool] | None:
        gi_file = self._root / ".gitignore"
        if not gi_file.exists():
            return None
        try:
            import pathspec  # type: ignore[import-not-found]
        except ImportError:
            logger.debug("pathspec not installed; skipping .gitignore rules")
            return None
        try:
            patterns = gi_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            return None
        spec = pathspec.PathSpec.from_lines("gitwildmatch", patterns)
        return spec.match_file
