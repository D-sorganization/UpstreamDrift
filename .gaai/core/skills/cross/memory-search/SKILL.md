---
name: memory-search
description: Search memory by frontmatter fields, full-text keywords, or cross-reference graph. Returns ranked file list — never loads full content. Use when the agent needs to find relevant memory without knowing exact paths.
license: MIT
compatibility: Works with any filesystem-based AI coding agent
metadata:
  author: gaai-framework
  version: "1.0"
  category: cross
  track: cross-cutting
  id: SKILL-CRS-024
  updated_at: 2026-03-01
  status: stable
  tags:
    - memory
    - search
    - retrieval
    - cross-reference
inputs:
  - search_mode: A | B | C
  - query: mode-specific (see Process)
  - contexts/memory/**  (read-only scan)
outputs:
  - search_results: list of {file_path, id, title, relevance, excerpt} (~2,000 tokens max)
---

# Memory Search

## Purpose / When to Activate

Activate when an agent needs to **find** relevant memory but does not know the exact file path, domain, or DEC ID.

This skill **locates** memory — it does not **load** it. After results are returned, the agent invokes `memory-retrieve` to load the specific files.

Use cases:
relevance: related_to_inbound # or: frontmatter_match | content_match | direct_mention | related_to_outbound
excerpt: "prevents connection exhaustion under load" # ~50 tokens max, absent in Mode A

```

Agent receives this list and decides which files to load via `memory-retrieve`.

---

## Non-Goals

This skill must NOT:
```
