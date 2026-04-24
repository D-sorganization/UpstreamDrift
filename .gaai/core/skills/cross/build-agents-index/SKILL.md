---
name: build-agents-index
description: Scan all agent and sub-agent definition files in .gaai/core/agents/, extract YAML frontmatter, merge with specialists.registry.yaml, and generate a derived agents-index.yaml at .gaai/core/agents/agents-index.yaml. Activate after adding, modifying, or removing any agent, sub-agent, or specialist entry.
license: MIT
compatibility: Works with any filesystem-based AI coding agent
metadata:
  author: gaai-framework
  version: "1.0"
  category: cross
  track: cross-cutting
  id: SKILL-CRS-018
  tags:
    - governance
    - index
    - discoverability
    - agents
  updated_at: 2026-02-26
  status: stable
inputs:
  - .gaai/core/agents/**                      (all agent .md files — scanned for frontmatter)
  - .gaai/core/agents/specialists.registry.yaml   (specialist definitions — included as-is)
outputs:
  - .gaai/core/agents/agents-index.yaml      (generated — never edit manually)
---

# Build Agents Index

## Purpose / When to Activate

Activate when:
