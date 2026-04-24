---
name: remediate-failures
description: Correct failures, rule violations, and acceptance criteria gaps detected during QA review. Activate when qa-review returns FAIL. Fixes without redefining scope — loops until all quality gates pass.
license: MIT
compatibility: Works with any filesystem-based AI coding agent
metadata:
  author: gaai-framework
  version: "1.0"
  category: delivery
  track: delivery
  id: SKILL-REMEDIATE-FAILURES-001
  updated_at: 2026-01-27
  status: stable
inputs:
  - qa_report  (failing)
  - contexts/artefacts/stories/**
  - contexts/artefacts/plans/**
  - contexts/rules/**
  - memory_context_bundle  (optional)
outputs:
  - updated_code_changes
  - remediation_notes
  - updated_qa_inputs  (for re-validation)
---

# Remediate Failures

## Purpose / When to Activate

Activate when:
