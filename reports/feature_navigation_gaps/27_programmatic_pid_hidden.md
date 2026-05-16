---
title: Programmatic P&ID Generator has no launcher presence
labels: feature, priority/P3
---

## Problem

The programmatic PID module (`src/shared/python/programmatic_pid/`) generates Piping & Instrumentation Diagrams with:

- Controls, equipment, instruments, layout, rendering
- Streams, title blocks, validation
- CLI entry point (`cli.py`)

This is a niche but complete tool with no way for users to discover it.

## Classification

**Borderline / Niche**: P&ID generation is tangential to biomechanics simulation. It may not warrant a standalone tile in a golf physics launcher, but could be useful for facility/process engineers.

## Acceptance Criteria

- [ ] Consider adding a `pid_generator` tile if there's user demand
- [ ] Alternative: document as an advanced tool accessible via CLI
- [ ] If added, assign to `developer_tools` category
