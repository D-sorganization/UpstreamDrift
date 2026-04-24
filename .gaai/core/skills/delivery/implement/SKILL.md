---
name: implement
description: Generate correct, minimal, maintainable code that satisfies a validated Story's acceptance criteria against an execution plan. Activate when a Story is validated, a plan exists, and all prerequisites are unambiguous.
license: MIT
compatibility: Works with any filesystem-based AI coding agent
metadata:
  author: gaai-framework
  version: "1.0"
  category: delivery
  track: delivery
  id: SKILL-IMPLEMENT-001
  updated_at: 2026-02-26
  status: stable
inputs:
  - contexts/artefacts/stories/**  (validated)
  - contexts/artefacts/plans/**
  - contexts/rules/**
  - memory_context_bundle
outputs:
  - code_changes
  - test_artifacts
  - implementation_report
---

# Implementation

## Purpose / When to Activate

Activate when:
