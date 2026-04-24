---
name: create-prd
description: Produce a lightweight strategic PRD that defines product vision, user problem, value hypothesis, success metrics, and scope boundaries. Activate only when starting a new product, launching a major initiative, or facing strategic uncertainty.
license: MIT
compatibility: Works with any filesystem-based AI coding agent
metadata:
  author: gaai-framework
  version: "1.0"
  category: discovery
  track: discovery
  id: SKILL-CREATE-PRD-001
  updated_at: 2026-01-27
  status: stable
inputs:
  - human_intent
  - core_user_problem
  - target_users
  - known_constraints (optional)
outputs:
  - contexts/artefacts/prd/*.md
---

# Create PRD

## Purpose / When to Activate

Activate when:
