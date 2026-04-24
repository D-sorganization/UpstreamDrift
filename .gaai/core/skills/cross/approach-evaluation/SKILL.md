---
name: approach-evaluation
description: Research industry standards and best practices, identify viable approaches for a given technical or architectural problem, and produce a structured factual comparison against project-specific constraints. Reports options — does not decide.
license: MIT
compatibility: Works with any filesystem-based AI coding agent
metadata:
  author: gaai-framework
  version: "1.0"
  category: cross
  track: cross-cutting
  id: SKILL-APPROACH-EVALUATION-001
  updated_at: 2026-02-26
  status: stable
inputs:
  - problem_statement                      (what needs to be solved)
  - contexts/memory/index.md               (registry — resolve project context, patterns, decisions files)
  - contexts/memory/**                     (categories resolved from index.md — project, patterns, decisions)
  - contexts/artefacts/stories/**          (the Story driving the evaluation, if in Delivery)
outputs:
  - contexts/artefacts/evaluations/{id}.approach-evaluation.md
---

# Approach Evaluation

## Purpose / When to Activate

Activate when the invoking agent identifies a technical or architectural decision point where:
