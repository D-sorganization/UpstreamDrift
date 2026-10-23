---
issue: 13648
summary: "Extract KnowledgePack search behind a compatibility module for repo-scoped query reuse."
dl_state: "in_review"
next_step: "Merge the search extraction while preserving current pin and query behavior."
branch: "codex/13648-knowledgepack-search"
paths: "src/shared/python/ai/knowledge/manifest.py,src/shared/python/ai/knowledge/pack.py,src/shared/python/ai/knowledge/search.py,tests/unit/ai/test_knowledge_pack_search.py"
---

Paired with [Tools PR #5722](https://github.com/D-sorganization/Tools/pull/5722) and issue #5718; this change extracts the existing search behavior without advancing the provider pin. Maintain the existing ai/knowledge seam ruling.

UD-PAIR: D-sorganization/Tools#5722
