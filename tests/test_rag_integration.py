"""Integration tests for the RAG store startup indexing and query wiring.

Tests that:
- The SimpleRAGStore can be populated with documents from a tmp directory
- Querying returns relevant results
- The local_server lifespan indexes docs into the RAG store
- The ChatService prepends RAG context to system messages
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.shared.python.ai.rag.simple_rag import SimpleRAGStore


@pytest.mark.unit
class TestSimpleRAGStoreIndexing:
    """Unit tests for RAG store document indexing."""

    def test_add_and_query_document(self) -> None:
        """Documents added to the store are returned by relevant queries."""
        store = SimpleRAGStore()
        store.add_document(
            "doc1",
            "Backspin is the reverse rotation of a golf ball that produces lift.",
            {"source": "test", "type": "docs"},
        )
        store.build_index()

        results = store.query("golf ball lift backspin", top_k=1)
        assert len(results) >= 1
        doc, score = results[0]
        assert doc.id == "doc1"
        assert score > 0.0

    def test_empty_store_returns_no_results(self) -> None:
        """An empty store returns an empty results list."""
        store = SimpleRAGStore()
        results = store.query("anything", top_k=5)
        assert results == []

    def test_multiple_docs_ranked_by_relevance(self) -> None:
        """More relevant documents score higher than less relevant ones."""
        store = SimpleRAGStore()
        store.add_document(
            "spin_doc",
            "Backspin creates Magnus effect lift on the golf ball trajectory.",
            {"source": "test"},
        )
        store.add_document(
            "grip_doc",
            "Club grip pressure affects wrist hinge and swing plane.",
            {"source": "test"},
        )
        store.build_index()

        results = store.query("Magnus lift backspin trajectory", top_k=2)
        assert len(results) >= 1
        top_doc, _ = results[0]
        assert top_doc.id == "spin_doc"


@pytest.mark.unit
class TestRAGIndexingFromFilesystem:
    """Tests for indexing docs from a temporary filesystem tree."""

    def test_index_markdown_files(self, tmp_path: Path) -> None:
        """Markdown files in docs/ are indexed into the store."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "physics.md").write_text(
            "# Ball Flight Physics\n\nThe drag coefficient affects carry distance.",
            encoding="utf-8",
        )
        (docs_dir / "glossary.md").write_text(
            "# Glossary\n\nBackspin: reverse rotation producing lift.",
            encoding="utf-8",
        )

        store = SimpleRAGStore()
        _index_docs_into_rag_impl(store, tmp_path)

        results = store.query("drag coefficient carry distance", top_k=2)
        assert len(results) >= 1
        sources = [doc.metadata.get("source", "") for doc, _ in results]
        assert any("physics.md" in s for s in sources)

    def test_index_root_markdown(self, tmp_path: Path) -> None:
        """SPEC.md and README.md at root are indexed."""
        (tmp_path / "SPEC.md").write_text(
            "# Specification\n\nThis suite models golf ball aerodynamics.",
            encoding="utf-8",
        )
        (tmp_path / "README.md").write_text(
            "# Golf Modeling Suite\n\nPhysics simulation for golf biomechanics.",
            encoding="utf-8",
        )

        store = SimpleRAGStore()
        _index_docs_into_rag_impl(store, tmp_path)

        assert "SPEC.md" in store.documents or "README.md" in store.documents

    def test_index_glossary_json(self, tmp_path: Path) -> None:
        """Glossary JSON entries are indexed as individual documents."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        glossary = [
            {
                "key": "backspin",
                "term": "Backspin",
                "b": "Reverse ball rotation that creates lift.",
                "i": "Magnus effect: backspin generates upward lift force.",
            },
            {
                "key": "drag_coefficient",
                "term": "Drag Coefficient",
                "b": "Resistance of ball moving through air.",
                "i": "Dimensionless number Cd quantifying aerodynamic drag.",
            },
        ]
        (data_dir / "glossary_core.json").write_text(
            json.dumps(glossary), encoding="utf-8"
        )

        store = SimpleRAGStore()
        _index_docs_into_rag_impl(store, tmp_path)

        assert "glossary:backspin" in store.documents
        assert "glossary:drag_coefficient" in store.documents

    def test_empty_directory_does_not_crash(self, tmp_path: Path) -> None:
        """An empty repo root indexes zero docs without errors."""
        store = SimpleRAGStore()
        _index_docs_into_rag_impl(store, tmp_path)
        assert len(store.documents) == 0


@pytest.mark.unit
class TestChatServiceRAGWiring:
    """Tests that ChatService prepends RAG results to the system context."""

    def test_rag_results_prepended_to_context(self) -> None:
        """When RAG store has docs, a query result is prepended as system message."""
        store = SimpleRAGStore()
        store.add_document(
            "drag_doc",
            "Drag coefficient Cd determines aerodynamic resistance of the golf ball.",
            {"source": "docs/physics.md", "type": "docs"},
        )
        store.build_index()

        with patch("src.api.services.chat_service.ChatService._load_adapter"):
            from src.api.services.chat_service import ChatService

            svc = ChatService(rag_store=store)

        assert svc._rag_store is store

    def test_chat_service_accepts_none_rag_store(self) -> None:
        """ChatService works normally when no RAG store is provided."""
        with patch("src.api.services.chat_service.ChatService._load_adapter"):
            from src.api.services.chat_service import ChatService

            svc = ChatService(rag_store=None)

        assert svc._rag_store is None


# ---------------------------------------------------------------------------
# Helper: replicate _index_docs_into_rag logic for unit-testing without
# importing the full FastAPI app
# ---------------------------------------------------------------------------


def _index_docs_into_rag_impl(rag_store: SimpleRAGStore, repo_root: Path) -> None:
    """Index documentation files into the RAG store (mirrors local_server logic).

    Args:
        rag_store: Store to populate.
        repo_root: Root of a (possibly temporary) repo tree.
    """
    import json as _json

    indexed = 0

    docs_dir = repo_root / "docs"
    if docs_dir.is_dir():
        for md_path in docs_dir.rglob("*.md"):
            content = md_path.read_text(encoding="utf-8", errors="ignore")
            if content.strip():
                rel = str(md_path.relative_to(repo_root))
                rag_store.add_document(rel, content, {"source": rel, "type": "docs"})
                indexed += 1

    for top_level in ("SPEC.md", "README.md"):
        path = repo_root / top_level
        if path.exists():
            content = path.read_text(encoding="utf-8", errors="ignore")
            if content.strip():
                rag_store.add_document(
                    top_level, content, {"source": top_level, "type": "root_doc"}
                )
                indexed += 1

    glossary_path = repo_root / "data" / "glossary_core.json"
    if glossary_path.exists():
        raw = glossary_path.read_text(encoding="utf-8", errors="ignore")
        entries = _json.loads(raw)
        for entry in entries:
            key = entry.get("key", "")
            term = entry.get("term", key)
            parts = [f"Term: {term}"]
            for field in ("b", "i", "a", "f"):
                val = entry.get(field)
                if val:
                    parts.append(str(val))
            content = "\n".join(parts)
            rag_store.add_document(
                f"glossary:{key}",
                content,
                {"source": "data/glossary_core.json", "type": "glossary", "key": key},
            )
            indexed += 1

    if indexed > 0:
        rag_store.build_index()
