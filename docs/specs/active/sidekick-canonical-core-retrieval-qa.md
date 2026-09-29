# Sidekick Canonical Core Retrieval Q&A

- status: active
- issue: [#6810](https://github.com/D-sorganization/UpstreamDrift/issues/6810)
- branch: `feat/6810-sidekick-retrieval-qa`

## Problem

Sidekick needs to answer setup questions about the Canonical Core without
guessing from broad chat context. The answer must be grounded in the local
Canonical Core conventions, schemas, adapter notes, and ADRs with source
citations.

## Scope

- Build a bounded local index over Canonical Core docs and schemas.
- Return deterministic extractive answers with `path:start-end` citations.
- Register the Q&A function as a read-only Sidekick chat tool.
- Keep retrieval local and free of command execution or file mutation.

## Non-goals

- General web search.
- Autonomous adapter edits or setup command execution.
- A replacement for the existing Sidekick chat, codemap, or RAG surfaces.

## Design

`src/shared/python/canonical_core/sidekick_retrieval_qa.py` owns the
repo-specific retrieval layer. It uses the existing Sidekick RAG store for local
TF-IDF search, but constrains the corpus to Canonical Core docs and schema files
and builds the final answer deterministically from retrieved excerpts.

`src/api/services/chat_service.py` registers the tool as
`answer_canonical_core_question`, so the existing Sidekick chat service can call
it without importing launcher or UI internals.

## Acceptance Criteria

- Indexing skips missing default corpus files and oversized files.
- Search returns ranked sources with stable file and line citations.
- Answers include citations and tell the model not to infer beyond sources.
- Empty questions and invalid result limits are rejected.
- The chat service exposes the tool in its default registry.

## Validation

- `python -m pytest tests/unit/canonical_core/test_sidekick_retrieval_qa.py`
- `python -m ruff check src/shared/python/canonical_core tests/unit/canonical_core src/api/services/chat_service.py`
- `python -m ruff format --check src/shared/python/canonical_core tests/unit/canonical_core src/api/services/chat_service.py`
