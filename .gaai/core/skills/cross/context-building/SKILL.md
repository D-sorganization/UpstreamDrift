---
name: context-building
description: Assemble a minimal, high-signal execution context bundle from already-retrieved memory, governed artefacts, and applicable rules. Activate after memory-retrieve and before any reasoning or execution skill.
license: MIT
compatibility: Works with any filesystem-based AI coding agent
metadata:
  author: gaai-framework
  version: "1.0"
  category: cross
  track: cross-cutting
  id: SKILL-CONTEXT-BUILDING-001
  updated_at: 2026-02-26
  status: stable
inputs:
  - retrieved_memory_bundle
  - contexts/artefacts/**  (relevant only)
  - contexts/rules/**  (applicable only)
outputs:
  - execution_context_bundle
---

# Context Building

## Purpose / When to Activate

Activate when context is fragmented or multiple memory sources need to be merged before a complex task:
