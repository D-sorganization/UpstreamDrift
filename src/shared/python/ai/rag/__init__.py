"""RAG (Retrieval-Augmented Generation) support for the AI assistant.

Provides indexing and simple retrieval for assistant knowledge.
"""

from src.shared.python.ai.rag.indexer_worker import IndexerWorker
from src.shared.python.ai.rag.simple_rag import SimpleRAGStore

__all__ = [
    "IndexerWorker",
    "SimpleRAGStore",
]
