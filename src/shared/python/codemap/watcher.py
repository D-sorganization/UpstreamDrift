# ruff: noqa: E501
"""``codemap-watch`` daemon — incremental re-index on file save.

Uses watchdog with a 500 ms debounce. On change, re-parses only the changed
file. Logs to ``<repo>/.codemap/watcher.log``.
"""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import sys
import threading
import time
from pathlib import Path

from . import api as api_mod
from . import db as db_mod
from . import indexer as indexer_mod
from . import parsers as parsers_mod

DEBOUNCE_S = 0.5


def _make_handler(repo: Path, conn, lock: threading.Lock, pending: dict, schedule):
    from watchdog.events import FileSystemEventHandler  # type: ignore[import-not-found]

    class _Handler(FileSystemEventHandler):
        def _enqueue(self, src_path: str, *, deleted: bool = False) -> None:
            try:
                abs_p = Path(src_path)
                if not deleted and not abs_p.is_file():
                    return
                if parsers_mod.language_for(abs_p) is None:
                    return
                try:
                    rel = abs_p.relative_to(repo).as_posix()
                except ValueError:
                    return
                if rel.startswith(".codemap/") or "/.codemap/" in rel:
                    return
                with lock:
                    pending[rel] = abs_p
                schedule()
            except Exception:  # pragma: no cover
                logging.getLogger("codemap.watcher").exception("enqueue failed")

        def on_modified(self, event):
            self._enqueue(event.src_path)

        def on_created(self, event):
            self._enqueue(event.src_path)

        def on_moved(self, event):
            self._enqueue(getattr(event, "dest_path", event.src_path))

        def on_deleted(self, event):
            self._enqueue(event.src_path, deleted=True)

    return _Handler()


def run(repo_root: str | None = None, *, debounce: float = DEBOUNCE_S) -> int:
    try:
        from watchdog.observers import Observer  # type: ignore[import-not-found]
    except Exception as exc:
        sys.stderr.write(f"codemap-watch requires watchdog: {exc}\n")
        return 2

    repo = Path(repo_root).resolve() if repo_root else api_mod.discover_repo_root()
    log_dir = repo / ".codemap"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("codemap.watcher")
    if not logger.handlers:
        handler = logging.handlers.RotatingFileHandler(
            log_dir / "watcher.log",
            maxBytes=512_000,
            backupCount=2,
        )
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.addHandler(logging.StreamHandler(sys.stderr))
        logger.setLevel(logging.INFO)

    conn = db_mod.open_db(repo)
    pending: dict[str, Path] = {}
    lock = threading.Lock()
    timer_ref: dict[str, threading.Timer | None] = {"t": None}

    def _flush() -> None:
        with lock:
            batch = dict(pending)
            pending.clear()
        if not batch:
            return
        logger.info("flushing %d change(s)", len(batch))
        for rel, abs_p in batch.items():
            try:
                if abs_p.exists():
                    stats = indexer_mod.RebuildStats()
                    indexer_mod._process_file(abs_p, rel, repo, conn, stats)  # type: ignore[attr-defined]
                else:
                    conn.execute("DELETE FROM symbols WHERE path = ?", (rel,))
                    conn.execute("DELETE FROM files WHERE path = ?", (rel,))
            except Exception:
                logger.exception("re-index failed for %s", rel)
        conn.commit()

    def _schedule() -> None:
        old = timer_ref.get("t")
        if old is not None:
            old.cancel()
        t = threading.Timer(debounce, _flush)
        t.daemon = True
        timer_ref["t"] = t
        t.start()

    handler = _make_handler(repo, conn, lock, pending, _schedule)
    observer = Observer()
    observer.schedule(handler, str(repo), recursive=True)
    observer.start()
    logger.info("codemap-watch started for %s", repo)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("codemap-watch stopping")
    finally:
        observer.stop()
        observer.join(timeout=5)
        _flush()
        conn.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="codemap-watch", description="Watch & re-index on save."
    )
    p.add_argument("--repo", default=None)
    p.add_argument("--debounce", type=float, default=DEBOUNCE_S)
    args = p.parse_args(argv)
    return run(args.repo, debounce=args.debounce)


if __name__ == "__main__":
    sys.exit(main())
