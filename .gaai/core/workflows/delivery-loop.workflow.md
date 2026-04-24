---
type: workflow
id: WORKFLOW-DELIVERY-LOOP-001
track: delivery
updated_at: 2026-02-23
---

# Delivery Loop Workflow

> **Branch model:** The delivery workflow targets the `staging` branch. AI never interacts with `production`. Promotion staging → production is a human action via GitHub PR.

## Purpose

Transform validated Stories into working, tested, governed software through coordinated sub-agent execution.

The Delivery Agent acts as orchestrator. It spawns specialized sub-agents, collects their handoff artefacts, and coordinates phase transitions until every Story either PASSes QA or ESCALATEs to the human.

---

## When to Use

- When Stories are validated and acceptance criteria are complete
- As the primary execution loop for all delivery work
- Invoked per Story or per batch from the active backlog

---

## Agent

**Delivery Agent / Orchestrator** (`agents/delivery.agent.md`)

Sub-agents spawned during execution:
