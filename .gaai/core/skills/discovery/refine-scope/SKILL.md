---
name: refine-scope
description: Iteratively refine Discovery artefacts (plans, epics, stories) when feedback, ambiguity, or uncertainty is detected. Activate when artefacts are incomplete, acceptance criteria are missing, or human feedback highlights gaps.
license: MIT
compatibility: Works with any filesystem-based AI coding agent
metadata:
  author: gaai-framework
  version: "1.0"
  category: discovery
  track: discovery
  id: SKILL-REFINE-SCOPE-001
  updated_at: 2026-01-28
  status: stable
inputs:
  - discovery_action_plan
  - contexts/artefacts/**  (partially completed)
  - contexts/memory/**  (selective)
  - human_feedback: numbered list of feedback items provided by the Discovery Agent, sourced from human input in the current session
outputs:
  - refined_discovery_plan
  - refined artefacts
---

# Refine Scope

## Purpose / When to Activate

Activate when:
