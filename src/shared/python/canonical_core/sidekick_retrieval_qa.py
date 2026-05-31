"""Deterministic Sidekick Q&A over Canonical Core docs and schemas.

This module deliberately does local retrieval and extractive answer building
only. It does not call an LLM, mutate repository state, or index arbitrary
paths by default.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.shared.python.ai.rag.simple_rag import SimpleRAGStore
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9_+#.-]+")
_MAX_FILE_BYTES = 500_000
_DEFAULT_TOP_K = 4
_DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_CORPUS = (
    Path("docs/conventions/canonical-v2.md"),
    Path("docs/adr/0026-canonical-dynamic-state-v2.md"),
    Path("docs/adr/0012-canonical-pose-interchange.md"),
    Path("docs/adr/0014-shared-biomech-models.md"),
    Path("docs/adr/0020-canonical-urdf-subsystem.md"),
    Path("docs/simulation_backends/results_schema_v2.md"),
    Path("docs/issues/backlog/036_simscape_adapter_protocol_skeleton.md"),
    Path("docs/issues/backlog/037_simscape_adapter_simulate.md"),
    Path("docs/issues/backlog/039_simscape_adapter_pool.md"),
    Path("src/shared/python/biomech/schemas/model_pack_v1.json"),
)


@dataclass(frozen=True)
class CanonicalCoreSource:
    """A cited source chunk returned by Canonical Core retrieval."""

    path: str
    start_line: int
    end_line: int
    score: float
    excerpt: str


@dataclass(frozen=True)
class CanonicalCoreAnswer:
    """Deterministic answer payload for Sidekick to display or rephrase."""

    question: str
    answer: str
    sources: tuple[CanonicalCoreSource, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": [asdict(source) for source in self.sources],
        }


@dataclass(frozen=True)
class _Chunk:
    doc_id: str
    path: str
    start_line: int
    end_line: int
    text: str


class CanonicalCoreRetrievalQA:
    """Bounded local retrieval over Canonical Core documentation.

    Args:
        repo_root: Repository root. Defaults to the current UpstreamDrift tree.
        corpus_paths: Relative paths to docs/schema files to index.
    """

    def __init__(
        self,
        repo_root: Path | None = None,
        corpus_paths: tuple[Path, ...] = _DEFAULT_CORPUS,
    ) -> None:
        self._repo_root = (repo_root or _DEFAULT_REPO_ROOT).resolve()
        self._corpus_paths = tuple(corpus_paths)
        self._store = SimpleRAGStore()
        self._chunks: dict[str, _Chunk] = {}
        self._indexed = False

    @property
    def document_count(self) -> int:
        """Return the number of indexed source chunks."""
        self._ensure_indexed()
        return len(self._chunks)

    def index(self) -> int:
        """Index configured docs and schemas.

        Returns:
            Number of chunks indexed.
        """
        self._store = SimpleRAGStore()
        self._chunks = {}
        for rel_path in self._corpus_paths:
            path = self._repo_root / rel_path
            if not path.is_file():
                logger.debug("Canonical Core retrieval source missing: %s", rel_path)
                continue
            if path.stat().st_size > _MAX_FILE_BYTES:
                logger.debug("Canonical Core retrieval source too large: %s", rel_path)
                continue
            self._index_file(path, rel_path.as_posix())
        self._indexed = True
        return len(self._chunks)

    def search(
        self,
        question: str,
        *,
        top_k: int = _DEFAULT_TOP_K,
    ) -> tuple[CanonicalCoreSource, ...]:
        """Return ranked source chunks for a question.

        Raises:
            ValueError: If ``question`` is empty or ``top_k`` is less than 1.
        """
        query = _validate_question(question)
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self._ensure_indexed()

        results = self._store.query(query, top_k=max(top_k * 4, top_k))
        if not results:
            return self._search_lexically(query, top_k=top_k)

        ranked: list[CanonicalCoreSource] = []
        for doc, vector_score in results:
            chunk = self._chunks.get(doc.id)
            if chunk is None:
                continue
            lexical_score = _lexical_score(query, chunk.text)
            combined = round(float(vector_score) + lexical_score, 6)
            if combined <= 0:
                continue
            ranked.append(
                CanonicalCoreSource(
                    path=chunk.path,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    score=combined,
                    excerpt=_normalize_excerpt(chunk.text),
                )
            )

        ranked.sort(key=lambda source: (-source.score, source.path, source.start_line))
        return tuple(ranked[:top_k])

    def answer(
        self,
        question: str,
        *,
        top_k: int = _DEFAULT_TOP_K,
    ) -> CanonicalCoreAnswer:
        """Build an extractive answer with source citations."""
        query = _validate_question(question)
        sources = self.search(query, top_k=top_k)
        if not sources:
            return CanonicalCoreAnswer(
                question=query,
                answer=(
                    "I could not find a Canonical Core source for that question in "
                    "the bounded docs/schema index. Ask about canonical-v2 units, "
                    "state layout, adapter boundaries, or model-pack schema fields."
                ),
                sources=(),
            )

        lines = [
            "Canonical Core guidance from the local docs/schema index:",
        ]
        for source in sources:
            cite = f"{source.path}:{source.start_line}-{source.end_line}"
            lines.append(f"- {source.excerpt} [{cite}]")
        lines.append(
            "Use these cited sources as grounding; do not infer behavior beyond them."
        )
        return CanonicalCoreAnswer(
            question=query,
            answer="\n".join(lines),
            sources=sources,
        )

    def _ensure_indexed(self) -> None:
        if not self._indexed:
            self.index()

    def _index_file(self, path: Path, rel_path: str) -> None:
        text = path.read_text(encoding="utf-8", errors="replace")
        for idx, chunk in enumerate(_chunk_text(text, rel_path), start=1):
            doc_id = f"{rel_path}#{idx}:{chunk.start_line}-{chunk.end_line}"
            indexed_chunk = _Chunk(
                doc_id=doc_id,
                path=rel_path,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                text=chunk.text,
            )
            self._chunks[doc_id] = indexed_chunk
            self._store.add_document(
                doc_id=doc_id,
                content=chunk.text,
                metadata={
                    "path": rel_path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "type": _source_type(path),
                },
            )

    def _search_lexically(
        self,
        question: str,
        *,
        top_k: int,
    ) -> tuple[CanonicalCoreSource, ...]:
        ranked: list[CanonicalCoreSource] = []
        for chunk in self._chunks.values():
            score = _lexical_score(question, chunk.text)
            if score <= 0:
                continue
            ranked.append(
                CanonicalCoreSource(
                    path=chunk.path,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    score=round(score, 6),
                    excerpt=_normalize_excerpt(chunk.text),
                )
            )
        ranked.sort(key=lambda source: (-source.score, source.path, source.start_line))
        return tuple(ranked[:top_k])


def answer_canonical_core_question(question: str, top_k: int = _DEFAULT_TOP_K) -> dict:
    """Answer a Canonical Core setup question for Sidekick.

    This is the read-only tool function registered by the chat service.
    """
    return CanonicalCoreRetrievalQA().answer(question, top_k=top_k).to_dict()


def _validate_question(question: str) -> str:
    if not isinstance(question, str):
        raise TypeError("question must be a string")
    cleaned = question.strip()
    if not cleaned:
        raise ValueError("question must be a non-empty string")
    if len(cleaned) > 1000:
        raise ValueError("question must be at most 1000 characters")
    return cleaned


def _source_type(path: Path) -> str:
    if path.suffix.lower() == ".json":
        return "schema"
    return "documentation"


def _tokenize(text: str) -> set[str]:
    return {match.group(0).lower() for match in _TOKEN_RE.finditer(text)}


def _lexical_score(query: str, text: str) -> float:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.0
    text_tokens = _tokenize(text)
    overlap = query_tokens & text_tokens
    return len(overlap) / len(query_tokens)


def _normalize_excerpt(text: str, max_chars: int = 360) -> str:
    compact = " ".join(line.strip() for line in text.splitlines() if line.strip())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def _chunk_text(text: str, rel_path: str) -> tuple[_Chunk, ...]:
    lines = text.splitlines()
    if rel_path.endswith(".json"):
        return _fixed_line_chunks(lines, rel_path, max_lines=40)
    return _markdown_chunks(lines, rel_path)


def _fixed_line_chunks(
    lines: list[str],
    rel_path: str,
    *,
    max_lines: int,
) -> tuple[_Chunk, ...]:
    chunks: list[_Chunk] = []
    for start_index in range(0, len(lines), max_lines):
        chunk_lines = lines[start_index : start_index + max_lines]
        text = "\n".join(chunk_lines).strip()
        if text:
            chunks.append(
                _Chunk(
                    doc_id="",
                    path=rel_path,
                    start_line=start_index + 1,
                    end_line=start_index + len(chunk_lines),
                    text=text,
                )
            )
    return tuple(chunks)


def _markdown_chunks(lines: list[str], rel_path: str) -> tuple[_Chunk, ...]:
    chunks: list[_Chunk] = []
    current: list[str] = []
    start_line = 1

    for line_no, line in enumerate(lines, start=1):
        if line.startswith("#") and current:
            _append_chunk(chunks, current, rel_path, start_line, line_no - 1)
            current = [line]
            start_line = line_no
        else:
            if not current:
                start_line = line_no
            current.append(line)
    if current:
        _append_chunk(chunks, current, rel_path, start_line, len(lines))

    expanded: list[_Chunk] = []
    for chunk in chunks:
        chunk_lines = chunk.text.splitlines()
        if len(chunk_lines) <= 45:
            expanded.append(chunk)
        else:
            fixed = _fixed_line_chunks(chunk_lines, rel_path, max_lines=35)
            for sub in fixed:
                expanded.append(
                    _Chunk(
                        doc_id="",
                        path=rel_path,
                        start_line=chunk.start_line + sub.start_line - 1,
                        end_line=chunk.start_line + sub.end_line - 1,
                        text=sub.text,
                    )
                )
    return tuple(expanded)


def _append_chunk(
    chunks: list[_Chunk],
    lines: list[str],
    rel_path: str,
    start_line: int,
    end_line: int,
) -> None:
    text = "\n".join(lines).strip()
    if text:
        chunks.append(
            _Chunk(
                doc_id="",
                path=rel_path,
                start_line=start_line,
                end_line=end_line,
                text=text,
            )
        )
