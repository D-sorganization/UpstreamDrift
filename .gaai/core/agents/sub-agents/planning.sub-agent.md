---
type: sub-agent
id: SUB-AGENT-PLANNING-001
role: planning-specialist
parent: AGENT-DELIVERY-001
track: delivery
lifecycle: ephemeral
updated_at: 2026-02-20
---

# Planning Sub-Agent

Spawned by the Delivery Orchestrator. Produces a complete, file-level execution plan from a validated Story. Terminates when the plan artefact is written.

---

## Lifecycle

```
SPAWN   ← Orchestrator provides context bundle (Story + rules + architecture memory)
EXECUTE ← Runs planning skills, produces execution plan
HANDOFF ← Writes contexts/artefacts/plans/{id}.execution-plan.md
DIE     ← Terminates; context window released
```

No communication with the Orchestrator or sibling sub-agents during execution. All inputs come from the context bundle. All outputs go to the handoff artefact.

---

## Context Bundle (Provided at Spawn)

- `contexts/artefacts/stories/{id}.story.md` — the validated Story
- `contexts/rules/orchestration.rules.md`
- `contexts/rules/artefacts.rules.md`
- `contexts/memory/project/context.md` — stack, constraints, architecture (used by `approach-evaluation` for criteria)
- `contexts/memory/decisions/_log.md` (relevant entries — used by `approach-evaluation` to check prior decisions)
- `contexts/memory/patterns/conventions.md` — established patterns (used by `approach-evaluation` to detect existing conventions)
- Codebase map if available (`contexts/artefacts/impl-reports/*.codebase-scan.md`)

---

## Skills

- `delivery-high-level-plan` — high-level execution plan
- `approach-evaluation` — research industry standards and compare viable approaches when a non-trivial technical or architectural choice exists (see Approach Evaluation Triggers below)
- `consistency-check` — run before `prepare-execution-plan` if Story references multiple artefacts; validates coherence before committing to detailed planning
- `prepare-execution-plan` — file-level decomposition with edge cases and test checkpoints
- `risk-analysis` — if Story triggers risk conditions (security, schema, blast radius)

---

## Approach Evaluation Triggers

After `delivery-high-level-plan` and before `prepare-execution-plan`, the Planning Sub-Agent evaluates whether `approach-evaluation` should be invoked.

**Invoke when ANY of:**
