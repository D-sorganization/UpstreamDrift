---
issue: 13648
summary: "Extract KnowledgePack BM25 search behind a private engine without changing its public query behavior."
dl_state: "in_review"
next_step: "Merge alongside Tools #5720 / PR #5719 to keep the paired ai/knowledge search extraction aligned."
branch: "codex/13648-knowledgepack-search"
paths: "src/shared/python/ai/knowledge/pack.py,src/shared/python/ai/knowledge/search.py,tests/unit/ai/test_knowledge_pack_search.py"
---

Paired with Tools #5720 / PR #5719; maintain the existing ai/knowledge seam ruling.
