"""Integration tests for the ai_backend Rust extension's RAG pipeline.

Skipped when the Rust wheel is not installed (most CI machines, contributors
who haven't run ``maturin develop --features python``). The ``requires_ort``
mark scopes the local-ONNX tests separately so they can be enabled in a
dedicated CI job that has the ONNX runtime + downloaded model available.
"""

from __future__ import annotations

import textwrap
import os
from pathlib import Path

import pytest


def _ai_backend_available() -> bool:
    try:
        import ai_backend  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name != "ai_backend":
            raise
        return False
    return True


_LOCAL_EMBEDDINGS_EXPECTED = os.environ.get("CI_LOCAL_EMBEDDINGS_EXPECTED") == "1"
_BACKEND_AVAILABLE = _ai_backend_available()
if _LOCAL_EMBEDDINGS_EXPECTED and not _BACKEND_AVAILABLE:
    raise RuntimeError(
        "CI_LOCAL_EMBEDDINGS_EXPECTED=1 but ai_backend is not importable"
    )

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        not _BACKEND_AVAILABLE,
        reason="ai_backend Rust extension not installed (run `maturin develop --features python` in rust_core/ai_backend)",
    ),
]


def test_memory_manager_cosine_search_returns_closest_first(tmp_path: Path) -> None:
    """Vector store ranks by cosine similarity, not insertion order."""
    from ai_backend import MemoryManager

    db = tmp_path / "mem.db"
    mem = MemoryManager(str(db))
    mem.initialize()

    # Two distinct unit vectors plus a vector very close to the query.
    mem.store_embedding("payload_far", [0.0, 1.0, 0.0])
    mem.store_embedding("payload_orth", [0.0, 0.0, 1.0])
    mem.store_embedding("payload_near", [0.9, 0.1, 0.05])
    mem.store_embedding("payload_exact", [1.0, 0.0, 0.0])

    hits = mem.search([1.0, 0.0, 0.0], 2)
    assert hits[0] == "payload_exact"
    assert hits[1] == "payload_near"


def test_memory_manager_is_idempotent(tmp_path: Path) -> None:
    """Re-inserting an identical payload should not duplicate."""
    from ai_backend import MemoryManager

    mem = MemoryManager(str(tmp_path / "dup.db"))
    mem.initialize()
    mem.store_embedding("only_one", [0.1, 0.2, 0.3])
    mem.store_embedding("only_one", [0.1, 0.2, 0.3])
    assert mem.count() == 1


@pytest.mark.requires_ort
@pytest.mark.skipif(
    not _LOCAL_EMBEDDINGS_EXPECTED,
    reason="dedicated local-embeddings lane is not enabled",
)
class TestLocalONNXEmbeddings:
    """Tests that exercise the on-device ONNX path. They:

    * use the pinned ~90 MB ONNX model (cached under
      ``$UPSTREAM_DRIFT_MODEL_CACHE`` or ``~/.cache/upstream-drift/models/``);
    * take ~2 s of inference per query on CPU.

    Run with ``pytest -m requires_ort`` to opt in.
    """

    def _make_pipeline(self, tmp_path: Path):
        from ai_backend import AIConfig, MemoryManager, RagPipeline

        cfg = AIConfig("", "http://unused", "unused-chat", str(tmp_path / "db"))
        mem = MemoryManager(str(tmp_path / "rag.db"))
        mem.initialize()
        rag = RagPipeline(mem, cfg, True)
        return rag, mem

    def test_repeat_indexing_dedupes_and_retrieves(self, tmp_path: Path) -> None:
        """Re-indexing the same file dedupes and the phrase retrieves its file.

        This checks retrieval and deduplication only. Embedding determinism is
        asserted directly at the Rust seam by ``max_embedding_drift`` against
        ``EMBEDDING_DETERMINISM_TOL`` (``rust_core/ai_backend/src/rag.rs``),
        including a nondeterministic substitute that must fail (#14288).
        """
        rag, mem = self._make_pipeline(tmp_path)

        text = "the quick brown fox jumps over the lazy dog"

        # Drop a fixture file and index it twice; dedupe means the chunk is
        # only stored once but the embedder runs.
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.py").write_text(text + "\n")
        n1 = rag.index_codebase(str(src))
        n2 = rag.index_codebase(str(src))
        assert n1 >= 1
        # Second index pass should be fully deduped.
        assert n2 == 0 or mem.count() == n1

        # Query with the same phrase: top-1 should be from `a.py`.
        hits = rag.retrieve_context(text, 1)
        assert len(hits) == 1
        assert "a.py" in hits[0]

    def test_embedding_similarity_orders_by_meaning(self, tmp_path: Path) -> None:
        """Semantically-related queries rank closer than unrelated ones."""
        rag, _mem = self._make_pipeline(tmp_path)

        src = tmp_path / "src"
        src.mkdir()
        (src / "pets.py").write_text("dog cat hamster pet rabbit\n")
        (src / "physics.py").write_text("kinetic energy momentum mass velocity\n")
        rag.index_codebase(str(src))

        pet_hits = rag.retrieve_context("which pet animals are mentioned", 1)
        physics_hits = rag.retrieve_context("conservation of momentum", 1)
        assert (
            pet_hits[0].endswith("dog cat hamster pet rabbit")
            or "pets.py" in pet_hits[0]
        )
        assert "physics.py" in physics_hits[0]

    def test_indexing_roundtrip_finds_known_phrase(self, tmp_path: Path) -> None:
        """A unique sentinel phrase should be retrievable verbatim."""
        rag, _mem = self._make_pipeline(tmp_path)

        sentinel = "xenosaurus_token_42 is the unique sentinel"
        fixture = tmp_path / "fixture"
        fixture.mkdir()
        for name, body in {
            "a.py": "def add(a, b): return a + b\n",
            "b.py": textwrap.dedent(f"""\
                # arbitrary file with a sentinel
                {sentinel}
                more lines below
                """),
            "c.py": "class Foo: pass\n",
        }.items():
            (fixture / name).write_text(body)

        n = rag.index_codebase(str(fixture))
        assert n >= 3
        hits = rag.retrieve_context(sentinel, 3)
        assert any("xenosaurus_token_42" in h for h in hits)
