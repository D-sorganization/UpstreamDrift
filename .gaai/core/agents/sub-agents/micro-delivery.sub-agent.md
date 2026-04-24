---
type: sub-agent
id: SUB-AGENT-MICRO-DELIVERY-001
role: micro-delivery-specialist
parent: AGENT-DELIVERY-001
track: delivery
lifecycle: ephemeral
updated_at: 2026-02-18
---

# MicroDelivery Sub-Agent

Spawned by the Delivery Orchestrator for low-complexity Stories (complexity ≤ 2). Handles plan + implement + QA in a single context window. Eliminates the overhead of three separate sub-agents for simple tasks.

---

## When the Orchestrator Spawns This Sub-Agent

```yaml
# Trigger conditions (all must be true):
complexity: ≤ 2
files_affected: ≤ 2
acceptance_criteria_count: ≤ 3
no_specialists_triggered: true # registry scan returns no matches
```

Typical tasks: bug fixes, typo corrections, single-line changes, dependency updates, rename operations, copy changes.

---

## Lifecycle

```
SPAWN   ← Orchestrator provides minimal context bundle
EXECUTE ← Plans, implements, and verifies in single context window
HANDOFF ← Writes combined contexts/artefacts/delivery/{id}.micro-delivery-report.md
DIE     ← Terminates; context window released
```

---

## Context Bundle (Provided at Spawn)

Deliberately minimal:
