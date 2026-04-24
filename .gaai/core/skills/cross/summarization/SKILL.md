---
name: summarization
description: Transform large, noisy, or short-term memory into compact, durable, high-signal summaries. Activate when session memory grows large, decisions accumulate, or memory retrieval starts returning too many files.
license: MIT
compatibility: Works with any filesystem-based AI coding agent
metadata:
  author: gaai-framework
  version: "1.0"
  category: cross
  track: cross-cutting
  id: SKILL-SUMMARIZATION-001
  updated_at: 2026-01-30
  status: stable
inputs:
  - contexts/memory/index.md        (registry — read first to discover all active categories)
  - contexts/memory/**              (any category registered in index.md — resolved at runtime)
outputs:
  - contexts/memory/summaries/*.summary.md
  - contexts/memory/archive/**
  - contexts/memory/index.md  (updated)
---

# Summarization

## Purpose / When to Activate

Activate when:
