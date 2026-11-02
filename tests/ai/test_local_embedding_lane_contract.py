from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.unit


def _load_rag_tests(monkeypatch, rag_pipeline, cache_path=None):
    backend = types.ModuleType("ai_backend")
    backend.AIConfig = lambda *args: object()
    backend.MemoryManager = Mock()
    backend.RagPipeline = rag_pipeline
    monkeypatch.setitem(sys.modules, "ai_backend", backend)
    monkeypatch.setenv("CI_LOCAL_EMBEDDINGS_EXPECTED", "1")
    monkeypatch.setenv("UPSTREAM_DRIFT_MODEL_OFFLINE", "1")
    if cache_path:
        monkeypatch.setenv("UPSTREAM_DRIFT_MODEL_CACHE", str(cache_path))
    path = Path(__file__).with_name("test_rust_rag_pipeline.py")
    spec = importlib.util.spec_from_file_location("rag_lane_contract_target", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_target(monkeypatch, tmp_path, pipeline, target):
    module = _load_rag_tests(monkeypatch, pipeline)
    getattr(module.TestLocalONNXEmbeddings(), target)(tmp_path)


def test_collection_is_model_free(tmp_path: Path, monkeypatch) -> None:
    model = Mock()
    _load_rag_tests(monkeypatch, model, tmp_path / "empty-cache")
    assert model.call_count == 0


def test_init_error_surfaces(tmp_path: Path, monkeypatch) -> None:
    broken = Mock(side_effect=RuntimeError("controlled initializer failure"))
    with pytest.raises(RuntimeError, match="controlled initializer failure"):
        _run_target(monkeypatch, tmp_path, broken, "_make_pipeline")


def test_missing_index_side_effect_fails(tmp_path: Path, monkeypatch) -> None:
    pipeline = Mock()
    pipeline.return_value.index_codebase.return_value = 0
    with pytest.raises(AssertionError):
        _run_target(
            monkeypatch,
            tmp_path,
            pipeline,
            "test_repeat_indexing_dedupes_and_retrieves",
        )


def test_bad_retrieval_output_fails(tmp_path: Path, monkeypatch) -> None:
    pipeline = Mock()
    pipeline.return_value.index_codebase.return_value = 1
    pipeline.return_value.retrieve_context.return_value = [
        "controlled unrelated output"
    ]
    with pytest.raises(AssertionError):
        _run_target(
            monkeypatch,
            tmp_path,
            pipeline,
            "test_embedding_similarity_orders_by_meaning",
        )
