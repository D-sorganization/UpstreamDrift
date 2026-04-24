---
name: memory-ingest
description: Transform validated knowledge into structured long-term memory. Activate after Bootstrap scan, after Discovery produces validated artefacts, or after architecture insights are available.
license: MIT
compatibility: Works with any filesystem-based AI coding agent
metadata:
  author: gaai-framework
  version: "1.0"
  category: cross
  track: cross-cutting
  id: SKILL-MEMORY-INGEST-001
  updated_at: 2026-02-26
  status: stable
inputs:
  - discovery_outputs  (validated)
  - architecture_insights
  - validated_decisions
  - project_knowledge
  - marketing_observation_logs  (validated hypotheses, promise drafts — from contexts/artefacts/marketing/**)
  - strategy_artefacts  (validated GTM decisions — from contexts/artefacts/strategy/**)
outputs:
  - contexts/memory/**  (any category registered in index.md)
  - contexts/memory/index.md  (updated)
---

# Memory Ingest

## Purpose / When to Activate

Activate after:
