from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _load_rag_tests(monkeypatch, rag_pipeline, cache_path=None):
    backend = types.ModuleType("ai_backend")
    backend.AIConfig = lambda *args: object()

    class MemoryManager:
        def __init__(self, *args):
            pass

        def initialize(self):
            pass

    backend.MemoryManager = MemoryManager
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


def test_collection_is_model_free(tmp_path: Path, monkeypatch) -> None:
    constructions = []

    class Recorder:
        def __init__(self, *args):
            constructions.append("model")

    _load_rag_tests(monkeypatch, Recorder, tmp_path / "empty-cache")
    assert constructions == []


def test_init_error_surfaces(tmp_path: Path, monkeypatch) -> None:
    class BrokenPipeline:
        def __init__(self, *args):
            raise RuntimeError("controlled initializer failure")

    module = _load_rag_tests(monkeypatch, BrokenPipeline)
    with pytest.raises(RuntimeError, match="controlled initializer failure"):
        module.TestLocalONNXEmbeddings()._make_pipeline(tmp_path)


def test_missing_index_side_effect_fails(tmp_path: Path, monkeypatch) -> None:
    class NoIndexPipeline:
        def __init__(self, *args):
            pass

        def index_codebase(self, path):
            return 0

    module = _load_rag_tests(monkeypatch, NoIndexPipeline)
    test = module.TestLocalONNXEmbeddings().test_repeat_indexing_dedupes_and_retrieves
    with pytest.raises(AssertionError):
        test(tmp_path)


def test_bad_retrieval_output_fails(tmp_path: Path, monkeypatch) -> None:
    class BadOutputPipeline:
        def __init__(self, *args):
            pass

        def index_codebase(self, path):
            return 1

        def retrieve_context(self, query, limit):
            return ["controlled unrelated output"]

    module = _load_rag_tests(monkeypatch, BadOutputPipeline)
    test = module.TestLocalONNXEmbeddings().test_embedding_similarity_orders_by_meaning
    with pytest.raises(AssertionError):
        test(tmp_path)
