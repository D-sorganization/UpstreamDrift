---
name: decision-extraction
description: Identify and formalize durable product and technical decisions from agent outputs into long-term memory. Activate after Discovery produces artefacts, Delivery resolves trade-offs, or product direction materially changes.
license: MIT
compatibility: Works with any filesystem-based AI coding agent
metadata:
  author: gaai-framework
  version: "1.0"
  category: cross
  track: cross-cutting
  id: SKILL-DECISION-EXTRACTION-001
  updated_at: 2026-03-03
  status: stable
inputs:
  - recent_agent_outputs: session outputs from the invoking agent, or file paths to artefacts produced in the current session (e.g., evaluation reports, refined stories, approach-evaluation outputs)
  - contexts/artefacts/**  (governed)
outputs:
  - contexts/memory/decisions/DEC-{N}.md  (individual ADR file)
  - contexts/memory/decisions/_log.md  (next ID updated)
  - contexts/memory/index.md  (registry + file count updated)
---

# Decision Extraction

## Purpose / When to Activate

Activate after:
supersedes: null # or DEC-{old-id} if replacing
superseded_by: null
tags:

- { relevant tags }
  related_to: [] # optional — max 5 DEC IDs

---

# DEC-{N} — Decision Title

## Context

---

## Decision

---

## Impact

```

---

## Quality Checks

- All major decisions become explicit memory
- No repeated reasoning across sessions
- Governance trail is traceable
- Memory grows only with high-signal knowledge

---

## Non-Goals

This skill must NOT:
```
