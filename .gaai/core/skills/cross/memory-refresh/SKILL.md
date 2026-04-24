---
name: memory-refresh
description: Periodic memory maintenance — archive session files, convert recurring knowledge into summaries, update the memory index. Activate at end of a major phase (Discovery complete, sprint done) or when memory spans many sessions. For emergency context-window pressure mid-task, use memory-compact instead.
license: MIT
compatibility: Works with any filesystem-based AI coding agent
metadata:
  author: gaai-framework
  version: "1.0"
  category: cross
  track: cross-cutting
  id: SKILL-MEMORY-REFRESH-001
  updated_at: 2026-03-03
  status: stable
inputs:
  - contexts/memory/index.md        (registry — read first to discover all active categories)
  - contexts/memory/**              (any category registered in index.md — resolved at runtime)
outputs:
  - contexts/memory/summaries/*.summary.md
  - contexts/memory/archive/**
  - contexts/memory/index.md  (updated)
---

# Memory Refresh

## Purpose / When to Activate

Activate:
