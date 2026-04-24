---
name: memory-index-sync
description: Detect and heal index.md drift — finds memory files on disk not registered in index.md and registers them. Run when /gaai-status reports unregistered files, after batch memory operations, or as a post-delivery gate.
license: MIT
compatibility: Works with any filesystem-based AI coding agent
metadata:
  author: gaai-framework
  version: "1.0"
  category: cross
  track: cross-cutting
  id: SKILL-MEMORY-INDEX-SYNC-001
  updated_at: 2026-03-03
  status: stable
inputs:
  - contexts/memory/  (full scan — read-only except index.md)
outputs:
  - contexts/memory/index.md  (registry updated if drift found)
  - sync_report  (inline summary of changes applied and anomalies flagged)
---

# Memory Index Sync

## Purpose / When to Activate

Activate when:
