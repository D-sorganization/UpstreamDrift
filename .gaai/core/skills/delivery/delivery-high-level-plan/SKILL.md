---
name: delivery-high-level-plan
description: Transform validated Stories into a clear, minimal, governed execution plan. Used by the Planning Sub-Agent as the first planning pass before prepare-execution-plan for Tier 2/3, or as the sole planning output for simple Stories.
license: MIT
compatibility: Works with any filesystem-based AI coding agent
metadata:
  author: gaai-framework
  version: "1.0"
  category: delivery
  track: delivery
  id: SKILL-DELIVERY-HIGH-LEVEL-PLAN-001
  updated_at: 2026-03-02
  owner: Planning Sub-Agent
  status: stable
inputs:
  - contexts/artefacts/stories/**  (validated)
  - acceptance_criteria
  - contexts/rules/**
  - contexts/memory/**  (selective)
  - technical_constraints  (optional)
outputs:
  - contexts/artefacts/plans/{id}.plan.md
---

# Delivery High-Level Execution Plan

## Purpose / When to Activate

**Owner: Planning Sub-Agent.** Not invoked directly by the Delivery Orchestrator.

Used by the Planning Sub-Agent as:
