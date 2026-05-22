"""RAG (Retrieval-Augmented Generation) context helpers.

Modules:
    context_provider: Builds LLM context from indexed document store.
    indexer_worker: Background worker for incremental document indexing.
    simple_rag: Lightweight single-file RAG over local markdown/text files.
"""

__all__ = [
    "context_provider",
    "indexer_worker",
    "simple_rag",
]
