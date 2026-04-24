---
name: memory-compact
description: Emergency single-pass memory compression when context window pressure is high mid-task. Activate when approaching token limits during active work. For scheduled end-of-phase cleanup, use memory-refresh instead.
license: MIT
compatibility: Works with any filesystem-based AI coding agent
metadata:
  author: gaai-framework
  version: "1.0"
  category: cross
  track: cross-cutting
  id: SKILL-MEMORY-COMPACT-001
  updated_at: 2026-03-01
  status: stable
inputs:
  - contexts/memory/index.md
  - contexts/memory/**/*
outputs:
  - contexts/memory/summaries/*.summary.md
  - contexts/memory/archive/**
  - contexts/memory/index.md  (updated)
---

# Memory Compact

## Purpose / When to Activate

Activate when:
