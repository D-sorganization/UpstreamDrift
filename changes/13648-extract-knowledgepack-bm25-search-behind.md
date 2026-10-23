---
issue: 13648
summary: "Characterize the existing KnowledgePack search contract before provider synchronization."
dl_state: "in_review"
next_step: "Use these regression tests when synchronizing the provider-owned search implementation."
branch: "codex/13648-knowledgepack-search"
paths: "tests/unit/ai/test_knowledge_pack_search.py"
---

Characterize current pinned KnowledgePack search behavior without modifying Tools-owned source. This tests BM25 order, FTS input sanitization, superseded filtering, and invalid inputs; the provider synchronization remains a separate follow-up after this test foundation merges.

Refs #13648
