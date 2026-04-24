---
name: codebase-scan
description: Create a high-level map of the project structure and identify architectural pillars, entry points, and module boundaries. Activate at Bootstrap initialization or before architecture extraction.
license: MIT
compatibility: Works with any filesystem-based AI coding agent
metadata:
  author: gaai-framework
  version: "1.0"
  category: cross
  track: cross-cutting
  id: SKILL-CODEBASE-SCAN-001
  updated_at: 2026-02-26
  status: stable
inputs:
  - repository/**
outputs:
  - codebase_tree
  - key_files_list
---

# Codebase Scan

## Purpose / When to Activate

Activate:
