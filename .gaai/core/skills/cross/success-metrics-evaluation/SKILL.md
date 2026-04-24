---
name: success-metrics-evaluation
description: Evaluate delivery outcomes against defined success metrics and acceptance goals. Activate after Delivery to verify that delivered work creates real business and technical impact, not just output.
license: MIT
compatibility: Works with any filesystem-based AI coding agent
metadata:
  author: gaai-framework
  version: "1.0"
  category: cross
  track: cross-cutting
  id: SKILL-SUCCESS-METRICS-EVALUATION-001
  updated_at: 2026-02-26
  status: future
inputs:
  - contexts/artefacts/stories/**
  - acceptance_criteria
  - delivered_artefacts
  - defined_success_metrics
  - runtime_or_usage_data  (optional)
outputs:
  - metric_results
  - story_level_success_report
  - gap_analysis
  - improvement_recommendations
---

# Success Metrics Evaluation

## Purpose / When to Activate

Activate after Delivery to verify outcomes, not just outputs. Prevents "output without outcome."

Use when:
