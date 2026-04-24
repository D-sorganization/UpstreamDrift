---
name: evaluate-story
description: Assess Story complexity, identify required domains, and determine delivery tier (MicroDelivery / Core Team / Core Team + Specialists). Activate as the first step of every delivery orchestration cycle.
license: MIT
compatibility: Works with any filesystem-based AI coding agent
metadata:
  author: gaai-framework
  version: "1.0"
  category: delivery
  track: delivery
  id: SKILL-DEL-007
  updated_at: 2026-02-26
  status: stable
inputs:
  - contexts/artefacts/stories/**       (the Story to evaluate)
  - agents/specialists.registry.yaml    (to check domain triggers)
  - core/skills/skills-index.yaml        (core framework skills)
  - project/skills/skills-index.yaml     (project-specific skills)
  - contexts/memory/index.md            (registry — resolve project context file from `project` category)
outputs:
  - evaluation result (inline — not written to file)
---

# Evaluate Story

## Purpose / When to Activate

Activate as the **first action** of every Delivery Orchestration cycle, before any sub-agent is spawned.

The Orchestrator needs to know:
