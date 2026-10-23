---
issue: 13648
summary: "Sync KnowledgePack search with merged Tools source while preserving query behavior."
dl_state: "in_review"
next_step: "Merge the synchronized KnowledgePack search source."
branch: "codex/13648-knowledgepack-search"
paths: "Cargo.toml,requirements-tools.txt,src/shared/python/ai/knowledge/manifest.py,src/shared/python/ai/knowledge/pack.py,src/shared/python/ai/knowledge/search.py,tests/unit/ai/test_knowledge_pack_search.py,vendor/ud-tools"
---

Paired with [Tools PR #5719](https://github.com/D-sorganization/Tools/pull/5719) and issue #5720; maintain the existing ai/knowledge seam ruling.
