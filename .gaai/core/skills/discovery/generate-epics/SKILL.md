---
name: generate-epics
description: Translate product intent or a PRD into a small set of outcome-driven Epics (3–7 max). Activate when starting a new product, adding a significant feature domain, or breaking down a PRD into actionable user outcomes.
license: MIT
compatibility: Works with any filesystem-based AI coding agent
metadata:
  author: gaai-framework
  version: "1.0"
  category: discovery
  track: discovery
  id: SKILL-GENERATE-EPICS-001
  updated_at: 2026-01-27
  status: stable
inputs:
  - product_intent  (or PRD if available)
outputs:
  - contexts/artefacts/epics/*.md
---

# Generate Epics

## Purpose / When to Activate

Activate when:
