"""Worker to index the codebase for RAG."""

import os
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

from src.shared.python.ai.rag.simple_rag import SimpleRAGStore
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


class IndexerWorker(QThread):
    """Background worker to index codebase."""

    progress = pyqtSignal(str)  # Status update
    finished = pyqtSignal(int)  # Emits number of docs indexed
    error = pyqtSignal(str)  # Emits error

    def __init__(self, root_path: Path, store: SimpleRAGStore) -> None:
        """Initialize worker.

        Args:
            root_path: Root directory to index.
            store: RAG store to populate.
        """
        super().__init__()
        self._root = root_path
        self._store = store

    def run(self) -> None:
        """Execute indexing."""
        try:
            self.progress.emit("Starting indexing...")
            count = 0

            # Simple walk
            for root, dirs, files in os.walk(self._root):
                # Filter directories
                dirs[:] = [
                    d
                    for d in dirs
                    if not d.startswith(".") and d != "__pycache__" and d != "venv"
                ]

                for file in files:
                    if file.endswith((".py", ".md", ".txt", ".json", ".xml", ".urdf")):
                        file_path = Path(root) / file

                        # Skip large files > 1MB
                        if file_path.stat().st_size > 1_000_000:
                            continue

                        try:
                            content = file_path.read_text(
                                encoding="utf-8", errors="ignore"
                            )
                            rel_path = str(file_path.relative_to(self._root))

                            self._store.add_document(
                                doc_id=rel_path,
                                content=content,
                                metadata={
                                    "path": str(file_path),
                                    "type": file_path.suffix,
                                },
                            )
                            count += 1
                            if count % 10 == 0:
                                self.progress.emit(f"Indexed {count} files...")
                        except (RuntimeError, ValueError, OSError) as e:
                            logger.warning(f"Failed to read {file_path}: {e}")

            self.progress.emit(f"Building TF-IDF index for {count} documents...")
            self._store.build_index()

            # Save index to a standard location
            index_path = self._root / ".antigravity_rag_index.json"
            self._store.save(index_path)

            self.finished.emit(count)

        except (FileNotFoundError, OSError) as e:
            logger.exception("Indexing failed")
            self.error.emit(str(e))
