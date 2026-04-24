---
name: architecture-extract
description: Convert raw project structure into clear architectural understanding — module boundaries, data flows, service relationships, and architectural patterns. Activate after codebase-scan during Bootstrap.
license: MIT
compatibility: Works with any filesystem-based AI coding agent
metadata:
  author: gaai-framework
  version: "1.0"
  category: cross
  track: cross-cutting
  id: SKILL-ARCHITECTURE-EXTRACT-001
  updated_at: 2026-02-26
  status: stable
inputs:
  - codebase_tree
  - key_files_list
outputs:
  - architecture_insights
---

# Architecture Extract

## Purpose / When to Activate

Activate:
