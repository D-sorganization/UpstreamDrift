"""Tests for Canonical Core retrieval Q&A used by Sidekick."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.shared.python.canonical_core.sidekick_retrieval_qa import (
    CanonicalCoreRetrievalQA,
    answer_canonical_core_question,
)


def _write_fixture_corpus(root: Path) -> tuple[Path, ...]:
    docs = root / "docs"
    schema_dir = root / "src" / "shared" / "python" / "biomech" / "schemas"
    docs.mkdir(parents=True)
    schema_dir.mkdir(parents=True)

    convention = docs / "canonical-v2.md"
    convention.write_text(
        "# canonical-v2\n"
        "\n"
        "## Units and frames\n"
        "Canonical Core uses SI units, a Z-up world frame, and gravity "
        "[0, 0, -9.80665] m/s^2.\n"
        "\n"
        "## State layout\n"
        "The state layout is q, v, a, and t. The base quaternion is ordered "
        "w, x, y, z and angular velocity is body-local.\n",
        encoding="utf-8",
    )

    adapter = docs / "adapter-guide.md"
    adapter.write_text(
        "# Adapter guide\n"
        "\n"
        "Adapters convert native engine state only at the adapter boundary. "
        "A setup path must cite the canonical-v2 table before converting "
        "quaternion order or angular velocity frames.\n",
        encoding="utf-8",
    )

    schema = schema_dir / "model_pack_v1.json"
    schema.write_text(
        "{\n"
        '  "$schema": "https://json-schema.org/draft/2020-12/schema",\n'
        '  "$id": "https://upstreamdrift.local/model_pack_v1.json",\n'
        '  "properties": {"schema": {"enum": ["model_pack/v1"]}}\n'
        "}\n",
        encoding="utf-8",
    )
    return (
        Path("docs/canonical-v2.md"),
        Path("docs/adapter-guide.md"),
        Path("src/shared/python/biomech/schemas/model_pack_v1.json"),
    )


def test_indexing_splits_docs_and_schema_with_line_citations(tmp_path: Path) -> None:
    corpus = _write_fixture_corpus(tmp_path)
    qa = CanonicalCoreRetrievalQA(repo_root=tmp_path, corpus_paths=corpus)

    assert qa.index() >= 3

    results = qa.search("What units and frame does canonical-v2 use?", top_k=2)
    assert results
    assert results[0].path == "docs/canonical-v2.md"
    assert results[0].start_line >= 1
    assert results[0].end_line >= results[0].start_line
    assert "SI units" in results[0].excerpt


def test_search_ranks_adapter_boundary_question_to_adapter_doc(
    tmp_path: Path,
) -> None:
    corpus = _write_fixture_corpus(tmp_path)
    qa = CanonicalCoreRetrievalQA(repo_root=tmp_path, corpus_paths=corpus)

    results = qa.search("How should adapters convert quaternion order?", top_k=2)

    assert results
    assert results[0].path == "docs/adapter-guide.md"
    assert "adapter boundary" in results[0].excerpt


def test_answer_includes_citations_and_no_autonomous_action(tmp_path: Path) -> None:
    corpus = _write_fixture_corpus(tmp_path)
    qa = CanonicalCoreRetrievalQA(repo_root=tmp_path, corpus_paths=corpus)

    answer = qa.answer("How do I set up canonical-v2 state layout?", top_k=2)

    assert "Canonical Core guidance" in answer.answer
    assert "docs/canonical-v2.md:" in answer.answer
    assert "do not infer behavior beyond them" in answer.answer
    assert answer.sources


def test_answer_reports_no_source_for_unmatched_question(tmp_path: Path) -> None:
    corpus = _write_fixture_corpus(tmp_path)
    qa = CanonicalCoreRetrievalQA(repo_root=tmp_path, corpus_paths=corpus)

    answer = qa.answer("zzzzzz_unknown_topic", top_k=2)

    assert answer.sources == ()
    assert "could not find" in answer.answer


def test_validation_rejects_empty_question(tmp_path: Path) -> None:
    qa = CanonicalCoreRetrievalQA(repo_root=tmp_path, corpus_paths=())

    with pytest.raises(ValueError, match="non-empty"):
        qa.search(" ")
    with pytest.raises(ValueError, match="top_k"):
        qa.search("canonical-v2", top_k=0)


def test_chat_service_registers_canonical_core_tool() -> None:
    from src.api.services.chat_service import ChatService

    service = ChatService()
    tool_names = {tool.name for tool in service._tool_registry.list_tools()}

    assert "answer_canonical_core_question" in tool_names


def test_tool_function_returns_serializable_payload() -> None:
    payload = answer_canonical_core_question("What is canonical-v2?", top_k=1)

    assert set(payload) == {"question", "answer", "sources"}
    assert isinstance(payload["sources"], list)
