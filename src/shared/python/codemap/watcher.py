"""Watchdog-based file watcher for incremental code-map updates.

When started, it monitors the repository tree for file-system changes and
calls :meth:`CodeMapIndex.update_file` on each modified source file.

Requires the ``watchdog`` package (``pip install watchdog``). Falls back
gracefully with a warning when watchdog is not installed.

Usage
-----
    from shared.python.codemap.watcher import start_watcher, stop_watcher

    stop_event = start_watcher(repo_root=Path("."))
    # ... application runs ...
    stop_watcher(stop_event)
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_INDEXABLE_EXTS = {".py", ".rs", ".ts", ".tsx", ".js", ".jsx", ".md", ".mdx"}


def start_watcher(
    repo_root: Path,
    *,
    db_path: Path | None = None,
) -> threading.Event:
    """Start a background watchdog thread for incremental index updates.

    Args:
        repo_root: Repository root to watch.
        db_path:   Optional override for the database path.

    Returns:
        A :class:`threading.Event` that can be set to stop the watcher.
        Pass it to :func:`stop_watcher`.
    """
    stop_event = threading.Event()

    try:
        from watchdog.events import (
            FileSystemEventHandler,  # type: ignore[import-untyped]
        )
        from watchdog.observers import Observer  # type: ignore[import-untyped]
    except ImportError:
        logger.warning(
            "watchdog not installed — incremental code-map updates disabled. "
            "Install with: pip install watchdog"
        )
        return stop_event

    from .indexer import CodeMapIndex

    class _Handler(FileSystemEventHandler):
        def __init__(self, index: CodeMapIndex) -> None:
            self._index = index

        def _handle(self, path_str: str) -> None:
            p = Path(path_str)
            if p.suffix.lower() in _INDEXABLE_EXTS:
                try:
                    n = self._index.update_file(p)
                    logger.debug("Updated index for %s (%d symbols)", p.name, n)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Watch update failed for %s: %s", p, exc)

        def on_modified(self, event: object) -> None:
            if not getattr(event, "is_directory", True):
                self._handle(getattr(event, "src_path", ""))

        def on_created(self, event: object) -> None:
            if not getattr(event, "is_directory", True):
                self._handle(getattr(event, "src_path", ""))

        def on_deleted(self, event: object) -> None:
            if not getattr(event, "is_directory", True):
                p = Path(getattr(event, "src_path", ""))
                if p.suffix.lower() in _INDEXABLE_EXTS:
                    try:
                        rel = p.relative_to(repo_root).as_posix()
                        from .db import delete_path

                        delete_path(self._index.conn, rel)
                        self._index.conn.commit()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "Failed to remove deleted file from index: %s", exc
                        )

    def _run() -> None:
        with CodeMapIndex(repo_root, db_path=db_path) as idx:
            handler = _Handler(idx)
            observer = Observer()
            observer.schedule(handler, str(repo_root), recursive=True)
            observer.start()
            logger.info("Code-map watcher started for %s", repo_root)
            try:
                stop_event.wait()
            finally:
                observer.stop()
                observer.join()
                logger.info("Code-map watcher stopped")

    t = threading.Thread(target=_run, daemon=True, name="codemap-watcher")
    t.start()
    return stop_event


def stop_watcher(stop_event: threading.Event) -> None:
    """Signal the watcher thread to shut down gracefully."""
    stop_event.set()
