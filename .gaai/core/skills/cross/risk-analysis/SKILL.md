---
name: risk-analysis
description: Systematically identify and structure product, delivery, and systemic risks before they become failures. Activate before finalizing Epics and Stories, before execution planning, after repeated QA failures, or after major scope changes.
license: MIT
compatibility: Works with any filesystem-based AI coding agent
metadata:
  author: gaai-framework
  version: "1.0"
  category: cross
  track: cross-cutting
  id: SKILL-RISK-ANALYSIS-001
  updated_at: 2026-01-27
  status: stable
inputs:
  - contexts/artefacts/**  (PRD, Epics, Stories, Plans as provided)
  - memory_context_bundle  (optional)
outputs:
  - contexts/artefacts/risk-reports/{story_id}.risk-report.md
  - flagged_risks  (structured list)
---

# Risk Analysis

## Purpose / When to Activate

Activate:
