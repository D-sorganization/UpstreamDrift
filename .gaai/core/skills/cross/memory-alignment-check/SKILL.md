---
name: memory-alignment-check
description: After QA PASS, compare the Story's implementation footprint against relevant memory entries. Reports confirmed entries, contradictions, and new knowledge candidates. Never writes to memory — produces a delta report for Discovery to action.
license: MIT
compatibility: Works with any filesystem-based AI coding agent
metadata:
  author: gaai-framework
  version: "1.0"
  category: cross
  track: cross-cutting
  id: SKILL-MEMORY-ALIGNMENT-CHECK-001
  updated_at: 2026-02-26
  status: stable
inputs:
  - contexts/artefacts/impl-reports/{id}.impl-report.md
  - contexts/artefacts/stories/{id}.story.md
  - contexts/memory/index.md
  - [selective memory entries by scope tags]
outputs:
  - contexts/artefacts/memory-deltas/{id}.memory-delta.md
---

# Memory Alignment Check

## Purpose / When to Activate

Activate after QA verdict is PASS — never before (avoids analysis on code that will change).

This skill checks that long-term memory (decisions, patterns, project context) remains accurate after a Story is delivered. It compares only the Story's implementation footprint against relevant memory entries — not the full codebase.

The codebase is the source of truth for implementation.
Memory is the source of truth for decisions and patterns.
This skill checks that both remain consistent after each delivery.

**This skill reports. It never writes to memory.**

---

## Process

### 1. Extract Implementation Footprint

From `{id}.impl-report.md`:

## Confirmed Entries

- memory_id: PATTERNS-001
  status: CONFIRMED
  suggested_last_verified_at: YYYY-MM-DD
  suggested_verified_against_story: E01S01
  note: Implementation of X is consistent with stated convention.

## Contradicted Entries

- memory_id: DEC-{N}
  status: CONTRADICTED
  severity: high | medium | low
  description: Memory states [X]. Implementation did [Y]. These are incompatible.
  action_required: Update or invalidate memory entry.

## New Knowledge Candidates

- candidate_id: CANDIDATE-001
  category: patterns | decisions | project
  description: New retry pattern introduced in services/api/client.ts — not yet in memory.
  suggested_tags: [api, resilience, patterns]
  ingestion_priority: high | medium | low

```

---

## Verdict Logic
```
