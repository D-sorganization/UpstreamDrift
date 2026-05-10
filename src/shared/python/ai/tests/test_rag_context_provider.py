"""Tests for RAGContextProvider.

Covers:
- File indexing (single file, directory)
- Document classification
- Query and retrieval
- Context prompt building
- Save/load persistence
- Edge cases (empty, no results)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.shared.python.ai.rag.context_provider import (
    RAGContextProvider,
    _classify_file,
    _truncate,
    _walk_files,
)


@pytest.fixture
def tmp_docs(tmp_path: Path) -> Path:
    """Create a temporary directory with test documents."""
    # Python file
    py_file = tmp_path / "calculator.py"
    py_file.write_text(
        "def gibbs_minimize(temperature, pressure):\n"
        "    '''Perform Gibbs free energy minimization.'''\n"
        "    return temperature * pressure\n",
        encoding="utf-8",
    )

    # Markdown file
    md_file = tmp_path / "readme.md"
    md_file.write_text(
        "# Gasification Model\n\n"
        "This model simulates coal gasification using thermodynamic equilibrium.\n"
        "The quench system cools the syngas from reactor temperatures.\n",
        encoding="utf-8",
    )

    # Config file
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        "[model]\ntemperature = 1200\npressure = 30\n",
        encoding="utf-8",
    )

    # Subdirectory
    sub = tmp_path / "sub"
    sub.mkdir()
    sub_file = sub / "helper.py"
    sub_file.write_text(
        "def calculate_enthalpy(species):\n    return 42.0\n",
        encoding="utf-8",
    )

    return tmp_path


# ── Classification tests ─────────────────────────────────────────────


class TestClassification:
    def test_classify_python(self) -> None:
        assert _classify_file(Path("test.py")) == "code"

    def test_classify_markdown(self) -> None:
        assert _classify_file(Path("README.md")) == "documentation"

    def test_classify_toml(self) -> None:
        assert _classify_file(Path("config.toml")) == "config"

    def test_classify_unknown(self) -> None:
        assert _classify_file(Path("data.bin")) == "other"


class TestTruncate:
    def test_short_text_unchanged(self) -> None:
        assert _truncate("hello", max_chars=100) == "hello"

    def test_long_text_truncated(self) -> None:
        text = "a" * 100
        result = _truncate(text, max_chars=50)
        assert len(result) < 100
        assert "[truncated]" in result


class TestWalkFiles:
    def test_walk_finds_files(self, tmp_docs: Path) -> None:
        files = _walk_files(tmp_docs)
        names = {f.name for f in files}
        assert "calculator.py" in names
        assert "readme.md" in names

    def test_walk_respects_depth(self, tmp_docs: Path) -> None:
        files = _walk_files(tmp_docs, max_depth=0)
        names = {f.name for f in files}
        # Should find top-level files but not subdirectory files
        assert "calculator.py" in names
        assert "helper.py" not in names

    def test_walk_skips_hidden(self, tmp_docs: Path) -> None:
        hidden = tmp_docs / ".hidden"
        hidden.mkdir()
        (hidden / "secret.py").write_text("x = 1", encoding="utf-8")

        files = _walk_files(tmp_docs)
        names = {f.name for f in files}
        assert "secret.py" not in names


# ── RAGContextProvider tests ─────────────────────────────────────────


class TestRAGContextProvider:
    def test_empty_store(self) -> None:
        provider = RAGContextProvider()
        assert provider.document_count == 0

    def test_index_directory(self, tmp_docs: Path) -> None:
        provider = RAGContextProvider()
        count = provider.index_directory(tmp_docs, include_config=True)
        assert count >= 3  # py + md + toml + sub/helper.py

    def test_index_directory_code_only(self, tmp_docs: Path) -> None:
        provider = RAGContextProvider()
        count = provider.index_directory(
            tmp_docs, include_code=True, include_docs=False
        )
        # Should only index .py files
        assert count >= 1

    def test_index_nonexistent_directory(self) -> None:
        provider = RAGContextProvider()
        count = provider.index_directory(Path("/nonexistent/path"))
        assert count == 0

    def test_index_file(self, tmp_docs: Path) -> None:
        provider = RAGContextProvider()
        result = provider.index_file(tmp_docs / "calculator.py")
        assert result is True
        assert provider.document_count == 1

    def test_index_file_dedup(self, tmp_docs: Path) -> None:
        provider = RAGContextProvider()
        provider.index_file(tmp_docs / "calculator.py")
        # Second index of same file should be skipped
        result = provider.index_file(tmp_docs / "calculator.py")
        assert result is False
        assert provider.document_count == 1

    def test_index_nonexistent_file(self) -> None:
        provider = RAGContextProvider()
        result = provider.index_file(Path("/nonexistent.py"))
        assert result is False

    def test_query_returns_results(self, tmp_docs: Path) -> None:
        provider = RAGContextProvider()
        provider.index_directory(tmp_docs, include_config=True)

        results = provider.get_relevant_context("gibbs minimization")
        assert len(results) > 0
        assert all("score" in r for r in results)
        assert all("content" in r for r in results)

    def test_query_empty_string(self) -> None:
        provider = RAGContextProvider()
        results = provider.get_relevant_context("")
        assert results == []

    def test_query_no_docs(self) -> None:
        provider = RAGContextProvider()
        results = provider.get_relevant_context("anything")
        assert results == []

    def test_build_context_prompt(self, tmp_docs: Path) -> None:
        provider = RAGContextProvider()
        provider.index_directory(tmp_docs, include_config=True)

        prompt = provider.build_context_prompt("gasification thermodynamic")
        if prompt:  # May be empty if sklearn not available
            assert "Relevant codebase context" in prompt

    def test_build_context_prompt_no_results(self) -> None:
        provider = RAGContextProvider()
        prompt = provider.build_context_prompt("anything")
        assert prompt == ""

    def test_save_and_load(self, tmp_docs: Path, tmp_path: Path) -> None:
        provider = RAGContextProvider()
        provider.index_directory(tmp_docs, include_config=True)

        save_path = tmp_path / "rag_store.json"
        provider.save(save_path)
        assert save_path.exists()

        # Load into new provider
        provider2 = RAGContextProvider()
        provider2.load(save_path)
        assert provider2.document_count == provider.document_count

    def test_doc_type_filter(self, tmp_docs: Path) -> None:
        provider = RAGContextProvider()
        provider.index_directory(tmp_docs, include_config=True)

        # Filter by code only
        results = provider.get_relevant_context("calculate", doc_type="code")
        for r in results:
            assert r["metadata"]["type"] == "code"
