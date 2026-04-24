---
name: post-mortem-learning
description: Analyze failures and suboptimal deliveries to identify root causes, contributing factors, and raw lessons. Activate after significant delivery failures, repeated QA failures, or when patterns of issues need to be understood.
license: MIT
compatibility: Works with any filesystem-based AI coding agent
metadata:
  author: gaai-framework
  version: "1.0"
  category: cross
  track: cross-cutting
  id: SKILL-POST-MORTEM-LEARNING-001
  updated_at: 2026-02-26
  status: experimental
inputs:
  - failed_or_degraded_story_results
  - qa_reports
  - contexts/artefacts/**  (delivered)
  - contexts/memory/decisions/**
  - contexts/rules/**  (applied)
outputs:
  - root_cause_analysis
  - contributing_factors
  - raw_lessons
  - failure_scenarios
  - improvement_candidates
---

# Post-Mortem Learning

## Purpose / When to Activate

Activate after:
