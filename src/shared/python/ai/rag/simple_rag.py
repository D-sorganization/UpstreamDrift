"""Simple RAG implementation using TF-IDF and Cosine Similarity.

This module provides a lightweight Retrieval Augmented Generation (RAG) system
that depends only on scikit-learn (which is already in the environment).
It enables the AI assistant to search relevant codebase snippets and documentation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

# Check for sklearn availability
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not found. RAG functionality will be limited.")


@dataclass
class Document:
    """A document in the RAG store."""

    id: str
    content: str
    metadata: dict[str, Any]


class SimpleRAGStore:
    """A simple vector store using TF-IDF."""

    def __init__(self) -> None:
        """Initialize empty store."""
        self.documents: dict[str, Document] = {}
        self.vectorizer: Any | None = None
        self.vectors: Any | None = None
        self._dirty = False  # Indicates if index needs rebuilding

    def add_document(
        self, doc_id: str, content: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Add a document to the store.

        Args:
            doc_id: Unique identifier.
            content: Text content.
            metadata: Optional metadata (path, type, etc).
        """
        self.documents[doc_id] = Document(
            id=doc_id,
            content=content,
            metadata=metadata or {},
        )
        self._dirty = True

    def remove_document(self, doc_id: str) -> None:
        """Remove a document."""
        if doc_id in self.documents:
            del self.documents[doc_id]
            self._dirty = True

    def build_index(self) -> None:
        """Build the TF-IDF index from current documents."""
        if not SKLEARN_AVAILABLE:
            return

        if not self.documents:
            return

        logger.info(f"Building RAG index with {len(self.documents)} documents...")

        # Collect all content
        self.doc_ids = list(self.documents.keys())
        corpus = [self.documents[did].content for did in self.doc_ids]

        # vectorizer
        self.vectorizer = TfidfVectorizer(stop_words="english")
        try:
            self.vectors = self.vectorizer.fit_transform(corpus)
            self._dirty = False
            logger.info("RAG index built successfully.")
        except ValueError:
            # Can happen if empty vocabulary (e.g. all stop words)
            logger.warning("Empty vocabulary in RAG index.")
            self.vectors = None

    def query(self, query_text: str, top_k: int = 5) -> list[tuple[Document, float]]:
        """Search for documents relevant to the query.

        Args:
            query_text: The search query.
            top_k: Number of results to return.

        Returns:
            List of (Document, score) tuples.
        """
        if not SKLEARN_AVAILABLE or not self.documents:
            return []

        if self._dirty:
            self.build_index()

        if self.vectors is None or self.vectorizer is None:
            return []

        # Transform query
        try:
            query_vec = self.vectorizer.transform([query_text])
        except ValueError:
            # Can happen if query contains only stop words or unknown words
            return []

        # Compute similarities
        similarities = cosine_similarity(query_vec, self.vectors).flatten()

        # Get top matching indices
        # argsort returns lowest to highest, so we reverse
        related_indices = similarities.argsort()[::-1]

        results = []
        for idx in related_indices[:top_k]:
            score = similarities[idx]
            if score > 0.0:  # Only return relevant results
                doc_id = self.doc_ids[idx]
                results.append((self.documents[doc_id], float(score)))

        return results

    def save(self, path: Path) -> None:
        """Save the store to disk."""
        data = {"documents": [asdict(doc) for doc in self.documents.values()]}

        # Save JSON
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path: Path) -> None:
        """Load store from disk."""
        if not path.exists():
            return

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        self.documents = {}
        for doc_data in data["documents"]:
            self.add_document(doc_data["id"], doc_data["content"], doc_data["metadata"])

        self._dirty = True  # Rebuild index on first query
