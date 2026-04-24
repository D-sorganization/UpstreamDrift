---
name: rules-normalize
description: Convert implicit or scattered project conventions into governed GAAI rule files, and create or modify rule files with integrity. Activate during Bootstrap, when creating a new rule, or when modifying an existing rule.
license: MIT
compatibility: Works with any filesystem-based AI coding agent
metadata:
  author: gaai-framework
  version: "1.0"
  category: cross
  track: cross-cutting
  id: SKILL-RULES-NORMALIZE-001
  updated_at: 2026-02-26
  status: stable
inputs:
  - detected_rule_files
  - existing_conventions  (linters, security configs, style guides, CI constraints)
  - project_guidelines
outputs:
  - contexts/rules/**
---

# Rules Normalize

## Purpose / When to Activate

Activate when:
