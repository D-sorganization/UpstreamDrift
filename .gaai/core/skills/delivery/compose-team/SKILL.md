---
name: compose-team
description: Assemble the context bundles for each sub-agent based on evaluate-story output. Produces spawn-ready packages for Planning, Implementation, QA, or MicroDelivery sub-agents. Activate after evaluate-story, before spawning any sub-agent.
license: MIT
compatibility: Works with any filesystem-based AI coding agent
metadata:
  author: gaai-framework
  version: "1.0"
  category: delivery
  track: delivery
  id: SKILL-DEL-008
  updated_at: 2026-02-18
  status: stable
inputs:
  - contexts/artefacts/stories/**         (the Story)
  - contexts/rules/**                     (applicable rules)
  - contexts/memory/index.md              (registry — resolve memory file paths before composing bundles)
  - contexts/memory/**                    (categories resolved from index.md at runtime)
  - agents/specialists.registry.yaml      (for Tier 3)
  - evaluate-story output                 (inline — tier + specialists_triggered)
outputs:
  - context bundles (inline — file lists passed to each sub-agent at spawn)
---

# Compose Team

## Purpose / When to Activate

Activate after `evaluate-story` returns the tier, before the first sub-agent is spawned.

The Orchestrator must give each sub-agent **exactly the context it needs — no more, no less**. Context pollution wastes tokens and introduces drift. Context starvation causes failures.

This skill determines what goes into each sub-agent's context bundle.

---

## Process

### Step 0 — Resolve memory file paths (always first)

Read `contexts/memory/index.md`. For each bundle below that references a memory category, resolve the actual file path from the index before including it. Never hardcode memory file paths — the index is the source of truth.

Key categories to resolve:
