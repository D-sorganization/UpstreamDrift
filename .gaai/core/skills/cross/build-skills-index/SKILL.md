---
name: build-skills-index
description: Scan SKILL.md files in .gaai/core/skills/ and .gaai/project/skills/, extract YAML frontmatter, and regenerate separate skills indices for each layer. Core index ships with the OSS framework; project index is project-specific.
license: MIT
compatibility: Works with any filesystem-based AI coding agent
metadata:
  author: gaai-framework
  version: "2.0"
  category: cross
  track: cross-cutting
  id: SKILL-CRS-017
  tags:
    - governance
    - index
    - discoverability
  updated_at: 2026-03-17
  status: stable
inputs:
  - .gaai/core/skills/**/SKILL.md       (core framework skills)
  - .gaai/project/skills/**/SKILL.md    (project-specific skills, if present)
outputs:
  - .gaai/core/skills/skills-index.yaml     (core skills only — ships with OSS)
  - .gaai/project/skills/skills-index.yaml  (project skills only — if project dir exists)
---

# Build Skills Index

## Purpose / When to Activate

Activate when:
