---
name: friction-retrospective
description: Scan delivery artefacts for friction log entries, detect recurring patterns, and produce retrospective reports. Invoked by Discovery Agent (never by Delivery) to identify systemic improvement opportunities from friction captured during delivery.
license: MIT
compatibility: Works with any filesystem-based AI coding agent
metadata:
  author: gaai-framework
  version: "1.0"
  category: cross
  track: cross-cutting
  id: SKILL-FRICTION-RETROSPECTIVE-001
  updated_at: 2026-03-01
  status: experimental
inputs:
  - contexts/artefacts/impl-reports/**
  - contexts/artefacts/qa-reports/**
  - contexts/artefacts/delivery/**
  - scope_filter (optional): epic, date_range, or friction_type
outputs:
  - retrospective_report (contexts/artefacts/retrospectives/{scope}.retro.md or inline)
---

# Friction Retrospective

## Purpose / When to Activate

Activate to aggregate and analyze friction captured during delivery. This skill reads `## Friction Log` sections from delivery artefacts and detects patterns that warrant promotion to durable memory (conventions, decisions, rule updates).

**Recommended triggers (conventions, not rules):**
