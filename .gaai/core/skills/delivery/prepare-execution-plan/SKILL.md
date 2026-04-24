---
name: prepare-execution-plan
description: Decompose a high-level delivery plan into a precise, file-level execution sequence with explicit ordering, edge cases, and test checkpoints. Activate after delivery-high-level-plan for complex or multi-phase Stories before implementation begins.
license: MIT
compatibility: Works with any filesystem-based AI coding agent
metadata:
  author: gaai-framework
  version: "1.0"
  category: delivery
  track: delivery
  id: SKILL-DEL-006
  updated_at: 2026-02-26
  status: stable
inputs:
  - contexts/artefacts/plans/*.plan.md        (from delivery-high-level-plan)
  - contexts/artefacts/stories/**             (validated)
  - contexts/rules/**
  - memory_context_bundle
  - codebase_map                              (optional, from codebase-scan)
outputs:
  - contexts/artefacts/plans/{id}.execution-plan.md
---

# Prepare Execution Plan

## Purpose / When to Activate

Activate after `delivery-high-level-plan` when the Story meets at least one of:
