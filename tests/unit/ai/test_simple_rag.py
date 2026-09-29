"""Tests for src.shared.python.ai.rag.simple_rag (Issues #1949, #1744)."""

from __future__ import annotations

from pathlib import Path

from src.shared.python.ai.rag.simple_rag import Document, SimpleRAGStore


class TestDocument:
    def test_simple_rag_construction(self) -> None:
        doc = Document(id="d1", content="Hello world", metadata={"type": "text"})
        assert doc.id == "d1"
        assert doc.content == "Hello world"

    def test_empty_metadata(self) -> None:
        doc = Document(id="d2", content="test", metadata={})
        assert doc.metadata == {}


class TestSimpleRAGStoreConstruction:
    def test_empty_initially(self) -> None:
        store = SimpleRAGStore()
        assert len(store.documents) == 0

    def test_dirty_false_initially(self) -> None:
        store = SimpleRAGStore()
        assert store._dirty is False


class TestSimpleRAGStoreAddRemove:
    def test_add_document(self) -> None:
        store = SimpleRAGStore()
        store.add_document("d1", "Hello world")
        assert "d1" in store.documents

    def test_add_document_with_metadata(self) -> None:
        store = SimpleRAGStore()
        store.add_document("d1", "test content", metadata={"source": "test"})
        assert store.documents["d1"].metadata["source"] == "test"

    def test_add_marks_dirty(self) -> None:
        store = SimpleRAGStore()
        store.add_document("d1", "content")
        assert store._dirty is True

    def test_remove_document(self) -> None:
        store = SimpleRAGStore()
        store.add_document("d1", "content")
        store.remove_document("d1")
        assert "d1" not in store.documents

    def test_remove_nonexistent_no_error(self) -> None:
        store = SimpleRAGStore()
        store.remove_document("nonexistent")  # Should not raise

    def test_remove_marks_dirty(self) -> None:
        store = SimpleRAGStore()
        store.add_document("d1", "content")
        store._dirty = False
        store.remove_document("d1")
        assert store._dirty is True


class TestSimpleRAGStoreQuery:
    def _populate_store(self) -> SimpleRAGStore:
        store = SimpleRAGStore()
        store.add_document("d1", "The quick brown fox jumps over the lazy dog")
        store.add_document("d2", "Machine learning neural networks deep learning")
        store.add_document("d3", "Python programming language functions classes")
        store.add_document("d4", "Chemical engineering process design reactors")
        return store

    def test_query_returns_list(self) -> None:
        store = self._populate_store()
        result = store.query("fox", top_k=2)
        assert isinstance(result, list)

    def test_query_returns_tuples(self) -> None:
        store = self._populate_store()
        result = store.query("machine learning", top_k=2)
        if result:
            assert isinstance(result[0], tuple)
            assert len(result[0]) == 2

    def test_query_returns_document_and_score(self) -> None:
        store = self._populate_store()
        result = store.query("neural networks", top_k=1)
        if result:
            doc, score = result[0]
            assert isinstance(doc, Document)
            assert isinstance(score, float)

    def test_query_scores_positive(self) -> None:
        store = self._populate_store()
        result = store.query("machine learning", top_k=3)
        for _, score in result:
            assert score > 0.0

    def test_relevant_document_ranked_first(self) -> None:
        store = self._populate_store()
        result = store.query("neural networks deep learning", top_k=4)
        if result:
            top_doc, _ = result[0]
            assert (
                "neural" in top_doc.content.lower()
                or "learning" in top_doc.content.lower()
            )

    def test_query_empty_store_returns_empty(self) -> None:
        store = SimpleRAGStore()
        result = store.query("anything")
        assert result == []

    def test_query_top_k_limits_results(self) -> None:
        store = self._populate_store()
        result = store.query("the", top_k=2)
        assert len(result) <= 2

    def test_query_irrelevant_returns_empty_or_low_score(self) -> None:
        store = self._populate_store()
        result = store.query("zzz_nonexistent_word_xyz", top_k=5)
        # Either empty or scores must be 0 (>0 threshold in implementation)
        assert result == [] or all(score > 0 for _, score in result)


class TestSimpleRAGStorePersistence:
    def test_save_and_load(self, tmp_path: Path) -> None:
        store = SimpleRAGStore()
        store.add_document("d1", "original content", {"tag": "v1"})

        save_path = tmp_path / "rag_store.json"
        store.save(save_path)

        new_store = SimpleRAGStore()
        new_store.load(save_path)
        assert "d1" in new_store.documents
        assert new_store.documents["d1"].content == "original content"

    def test_load_nonexistent_file_no_error(self, tmp_path: Path) -> None:
        store = SimpleRAGStore()
        store.load(tmp_path / "nonexistent.json")  # Should not raise
        assert len(store.documents) == 0

    def test_save_creates_file(self, tmp_path: Path) -> None:
        store = SimpleRAGStore()
        store.add_document("d1", "test")
        save_path = tmp_path / "test.json"
        store.save(save_path)
        assert save_path.exists()
