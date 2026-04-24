---
name: discovery-high-level-plan
description: Transform vague or high-level human intent into a governed Discovery action plan. Activate when intent is unclear, broad, or when multiple discovery steps are required before any artefact is created.
license: MIT
compatibility: Works with any filesystem-based AI coding agent
metadata:
  author: gaai-framework
  version: "1.0"
  category: discovery
  track: discovery
  id: SKILL-DISCOVERY-HIGH-LEVEL-PLAN-001
  updated_at: 2026-02-26
  status: stable
inputs:
  - human_intent
  - contexts/artefacts/**  (optional)
  - contexts/memory/**  (selective)
outputs:
  - discovery_action_plan
---

# Discovery High-Level Planning

## Purpose / When to Activate

Activate this skill when:
