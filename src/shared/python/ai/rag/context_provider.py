"""RAG context provider for chat sessions.

Wraps the existing ``SimpleRAGStore`` to provide contextual document
retrieval that enriches chat prompts with relevant codebase and
documentation snippets.

Supports:
- Auto-indexing from configured directories
- Per-query context injection into system prompts
- Document type filtering (code, docs, config)
- Relevance scoring

Usage::

    from src.shared.python.ai.rag.context_provider import RAGContextProvider

    provider = RAGContextProvider()
    provider.index_directory(Path("docs/"))
    context = provider.get_relevant_context("How does Gibbs minimization work?")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.shared.python.ai.rag.simple_rag import SimpleRAGStore

logger = logging.getLogger(__name__)

# File extensions to index by type
_CODE_EXTENSIONS = frozenset({".py", ".rs", ".ts", ".js", ".cpp", ".h"})
_DOC_EXTENSIONS = frozenset({".md", ".rst", ".txt", ".adoc"})
_CONFIG_EXTENSIONS = frozenset({".toml", ".yaml", ".yml", ".json", ".cfg", ".ini"})

# Maximum file size to index [bytes]
MAX_FILE_SIZE = 500_000  # 500 KB


class RAGContextProvider:
    """Provides RAG-augmented context for chat sessions.

    Wraps SimpleRAGStore with higher-level features: directory indexing,
    file type classification, and chat prompt enrichment.

    Attributes:
        store: The underlying RAG store.
    """

    def __init__(self, store: SimpleRAGStore | None = None) -> None:
        """Initialize RAG context provider.

        Args:
            store: Optional pre-built store. Creates a new one if None.
        """
        self._store = store or SimpleRAGStore()
        self._indexed_paths: set[str] = set()

    @property
    def store(self) -> SimpleRAGStore:
        """Return the underlying RAG store."""
        return self._store

    @property
    def document_count(self) -> int:
        """Return the number of indexed documents."""
        return len(self._store.documents)

    def index_directory(
        self,
        directory: Path,
        *,
        include_code: bool = True,
        include_docs: bool = True,
        include_config: bool = False,
        max_depth: int = 5,
    ) -> int:
        """Index all relevant files in a directory.

        Args:
            directory: Root directory to index.
            include_code: Index source code files.
            include_docs: Index documentation files.
            include_config: Index configuration files.
            max_depth: Maximum directory depth to traverse.

        Returns:
            Number of files indexed.
        """
        if not directory.exists():
            logger.warning("RAG index directory not found: %s", directory)
            return 0

        allowed_extensions: set[str] = set()
        if include_code:
            allowed_extensions |= _CODE_EXTENSIONS
        if include_docs:
            allowed_extensions |= _DOC_EXTENSIONS
        if include_config:
            allowed_extensions |= _CONFIG_EXTENSIONS

        count = 0
        for path in _walk_files(directory, max_depth=max_depth):
            if path.suffix.lower() not in allowed_extensions:
                continue

            if path.stat().st_size > MAX_FILE_SIZE:
                continue

            str_path = str(path.resolve())
            if str_path in self._indexed_paths:
                continue

            try:
                content = path.read_text(encoding="utf-8", errors="replace")
                doc_type = _classify_file(path)
                self._store.add_document(
                    doc_id=str_path,
                    content=content,
                    metadata={
                        "path": str_path,
                        "name": path.name,
                        "type": doc_type,
                        "extension": path.suffix,
                        "size_bytes": path.stat().st_size,
                    },
                )
                self._indexed_paths.add(str_path)
                count += 1
            except (OSError, UnicodeDecodeError):
                logger.debug("Skipping unreadable file: %s", path)
                continue

        if count > 0:
            logger.info(
                "Indexed %d files from %s (%d total documents)",
                count,
                directory,
                self.document_count,
            )

        return count

    def index_file(self, path: Path, doc_type: str | None = None) -> bool:
        """Index a single file.

        Args:
            path: File to index.
            doc_type: Optional document type override.

        Returns:
            True if indexed, False if skipped.
        """
        if not path.exists() or not path.is_file():
            return False

        str_path = str(path.resolve())
        if str_path in self._indexed_paths:
            return False

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            self._store.add_document(
                doc_id=str_path,
                content=content,
                metadata={
                    "path": str_path,
                    "name": path.name,
                    "type": doc_type or _classify_file(path),
                    "extension": path.suffix,
                },
            )
            self._indexed_paths.add(str_path)
            return True
        except (OSError, UnicodeDecodeError):
            return False

    def get_relevant_context(
        self,
        query: str,
        *,
        top_k: int = 5,
        min_score: float = 0.05,
        doc_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve documents relevant to a query.

        Args:
            query: User's question or message.
            top_k: Maximum results to return.
            min_score: Minimum relevance score (0.0-1.0).
            doc_type: Optional filter by document type.

        Returns:
            List of dicts with keys: content, score, metadata.
        """
        if not query or not query.strip():
            return []

        results = self._store.query(query, top_k=top_k * 2)  # Over-fetch for filtering

        filtered = []
        for doc, score in results:
            if score < min_score:
                continue
            if doc_type and doc.metadata.get("type") != doc_type:
                continue
            filtered.append(
                {
                    "content": _truncate(doc.content, max_chars=2000),
                    "score": round(score, 4),
                    "metadata": doc.metadata,
                }
            )
            if len(filtered) >= top_k:
                break

        return filtered

    def build_context_prompt(
        self,
        query: str,
        *,
        top_k: int = 3,
        min_score: float = 0.1,
    ) -> str:
        """Build a context string to inject into the system prompt.

        Args:
            query: User's question.
            top_k: Number of context snippets.
            min_score: Minimum relevance.

        Returns:
            Formatted context string, or empty string if no relevant docs.
        """
        results = self.get_relevant_context(query, top_k=top_k, min_score=min_score)

        if not results:
            return ""

        parts = ["Relevant codebase context:"]
        for i, r in enumerate(results, 1):
            name = r["metadata"].get("name", "unknown")
            score = r["score"]
            snippet = r["content"][:500]
            parts.append(f"\n--- [{i}] {name} (relevance: {score:.2f}) ---\n{snippet}")

        return "\n".join(parts)

    def save(self, path: Path) -> None:
        """Save the RAG store to disk.

        Args:
            path: File path to save to.
        """
        self._store.save(path)

    def load(self, path: Path) -> None:
        """Load the RAG store from disk.

        Args:
            path: File path to load from.
        """
        self._store.load(path)
        self._indexed_paths = set(self._store.documents.keys())


# ── Helpers ──────────────────────────────────────────────────────────


def _classify_file(path: Path) -> str:
    """Classify a file by its extension.

    Args:
        path: File to classify.

    Returns:
        Document type string.
    """
    ext = path.suffix.lower()
    if ext in _CODE_EXTENSIONS:
        return "code"
    if ext in _DOC_EXTENSIONS:
        return "documentation"
    if ext in _CONFIG_EXTENSIONS:
        return "config"
    return "other"


def _truncate(text: str, max_chars: int = 2000) -> str:
    """Truncate text to max_chars with ellipsis.

    Args:
        text: Text to truncate.
        max_chars: Maximum characters.

    Returns:
        Truncated text.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated]"


def _walk_files(directory: Path, max_depth: int = 5) -> list[Path]:
    """Walk directory up to max_depth, yielding files.

    Args:
        directory: Root directory.
        max_depth: Maximum recursion depth.

    Returns:
        List of file paths.
    """
    files: list[Path] = []
    _walk_recursive(directory, files, current_depth=0, max_depth=max_depth)
    return files


def _walk_recursive(
    directory: Path,
    files: list[Path],
    current_depth: int,
    max_depth: int,
) -> None:
    """Recursive directory walker."""
    if current_depth > max_depth:
        return

    try:
        for entry in sorted(directory.iterdir()):
            # Skip hidden and common ignored directories
            if entry.name.startswith(".") or entry.name in {
                "__pycache__",
                "node_modules",
                ".git",
                ".venv",
                "venv",
                "build",
                "dist",
            }:
                continue

            if entry.is_file():
                files.append(entry)
            elif entry.is_dir():
                _walk_recursive(entry, files, current_depth + 1, max_depth)
    except PermissionError:
        pass
