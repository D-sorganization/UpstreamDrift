"""RAG indexing controller for the AI assistant.

Owns the IndexerWorker lifecycle and reports status/messages back via
Qt signals so the panel can update its UI without coupling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal

from src.shared.python.ai.rag.indexer_worker import IndexerWorker
from src.shared.python.ai.rag.simple_rag import SimpleRAGStore
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


def _resolve_src_path() -> Path:
    repo_root = Path(__file__).resolve().parent  # gui
    while repo_root.name != "src" and repo_root.parent != repo_root:
        repo_root = repo_root.parent
    return repo_root if repo_root.name == "src" else Path("src").resolve()


class IndexingController(QObject):
    """Drives codebase indexing and exposes its lifecycle as signals."""

    status_changed = pyqtSignal(str)
    system_message = pyqtSignal(str)
    finished = pyqtSignal(int)
    failed = pyqtSignal(str)

    def __init__(self, rag_store: SimpleRAGStore, parent: Any = None) -> None:
        super().__init__(parent)
        self._rag_store = rag_store
        self._worker: IndexerWorker | None = None

    @property
    def is_running(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    @property
    def worker(self) -> IndexerWorker | None:
        return self._worker

    def start(self) -> None:
        """Kick off (or no-op if already running) a codebase index build."""
        if self.is_running:
            self.status_changed.emit("Indexing already in progress...")
            return
        self.status_changed.emit("Indexing codebase...")
        self.system_message.emit("Indexing local codebase context...")

        src_path = _resolve_src_path()
        if not src_path.exists():
            logger.error("Could not find src directory to index at %s", src_path)
            self.status_changed.emit("Error: 'src' not found")
            self.system_message.emit("Codebase indexing failed: 'src' not found.")
            return

        worker = IndexerWorker(src_path, self._rag_store)
        worker.progress.connect(self.status_changed.emit)
        worker.finished.connect(self._on_finished)
        worker.error.connect(self._on_error)
        self._worker = worker
        worker.start()

    def _on_finished(self, docs_indexed: int) -> None:
        self.status_changed.emit(f"Index ready ({docs_indexed} docs)")
        self.system_message.emit(f"Codebase index ready ({docs_indexed} docs indexed).")
        self._worker = None
        self.finished.emit(docs_indexed)

    def _on_error(self, error: str) -> None:
        self.status_changed.emit(f"Index error: {error}")
        self.system_message.emit(f"Codebase indexing failed: {error}")
        self._worker = None
        self.failed.emit(error)
