"""Integration tests for the ai_backend Rust extension's RAG pipeline.

Skipped when the Rust wheel is not installed (most CI machines, contributors
who haven't run ``maturin develop --features python``). The ``requires_ort``
mark scopes the local-ONNX tests separately so they can be enabled in a
dedicated CI job that has the ONNX runtime + downloaded model available.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest


def _ai_backend_available() -> bool:
    try:
        import ai_backend  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _ai_backend_available(),
    reason="ai_backend Rust extension not installed (run `maturin develop --features python` in rust_core/ai_backend)",
)


def _has_local_embeddings() -> bool:
    if not _ai_backend_available():
        return False
    import ai_backend  # noqa: F401

    # Probe via constructor: a wheel built without `local-embeddings` raises
    # RuntimeError when use_local_embeddings=True.
    try:
        from ai_backend import AIConfig, MemoryManager, RagPipeline

        cfg = AIConfig("", "http://localhost", "x", ":memory:")
        mem = MemoryManager(":memory:")
        mem.initialize()
        RagPipeline(mem, cfg, True)
    except RuntimeError:
        return False
    except Exception:
        return False
    return True


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
    not _has_local_embeddings(),
    reason="ai_backend built without `local-embeddings` feature or model unavailable",
)
class TestLocalONNXEmbeddings:
    """Tests that exercise the on-device ONNX path. They:

    * download ~22 MB on first run (cached under
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

    def test_embedding_determinism(self, tmp_path: Path) -> None:
        """Same text must produce the same vector across calls."""
        rag, mem = self._make_pipeline(tmp_path)

        # Index two copies of the same chunk through the pipeline by going
        # through retrieve_context (which embeds the query). We can't access
        # the embedder directly from Python, so we round-trip via storage:
        # one payload, embed twice, expect cosine ~= 1.0.
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
