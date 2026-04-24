---
name: consistency-check
description: Detect inconsistencies across related artefacts and governance constraints. Activate after story generation, after plan preparation, before implementation, or after remediation attempts. Reports issues — does not fix them.
license: MIT
compatibility: Works with any filesystem-based AI coding agent
metadata:
  author: gaai-framework
  version: "1.0"
  category: cross
  track: cross-cutting
  id: SKILL-CONSISTENCY-CHECK-001
  updated_at: 2026-01-30
  status: stable
inputs:
  - contexts/artefacts/**  (Epics, Stories, Plans, PRDs as applicable)
  - contexts/rules/**
  - memory_context_bundle  (optional)
outputs:
  - contexts/artefacts/consistency-reports/{story_id}.consistency-report.md
  - flagged_issues  (structured list)
---

# Consistency Check

## Purpose / When to Activate

Activate:
