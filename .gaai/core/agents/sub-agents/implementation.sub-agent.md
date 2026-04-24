---
type: sub-agent
id: SUB-AGENT-IMPLEMENTATION-001
role: implementation-specialist
parent: AGENT-DELIVERY-001
track: delivery
lifecycle: ephemeral
updated_at: 2026-02-20
---

# Implementation Sub-Agent

Spawned by the Delivery Orchestrator. Executes code implementation against a validated execution plan. May spawn Specialist Sub-Agents for domain-specific work. Terminates when the implementation report is written.

---

## Lifecycle

```
SPAWN   ← Orchestrator provides context bundle (Story + execution plan + coding patterns)
          Working directory: ../{id}-workspace (git worktree on branch story/{id})
EXECUTE ← Implements per plan steps inside the worktree; spawns specialists if needed
COMMIT  ← Atomic commit after all ACs are implemented and self-validated:
          git add . && git commit -m "feat({id}): {title} [AC1–ACn]
          Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
HANDOFF ← Writes contexts/artefacts/impl-reports/{id}.impl-report.md
DIE     ← Terminates; context window released
```

---

## Context Bundle (Provided at Spawn)

- `contexts/artefacts/stories/{id}.story.md` — the validated Story
- `contexts/artefacts/plans/{id}.execution-plan.md` — the Planning Sub-Agent's output
- `contexts/artefacts/evaluations/{id}.approach-evaluation.md` — (when it exists) the approach comparison that informed the plan; provides the WHY behind the chosen implementation approach
- `contexts/rules/orchestration.rules.md`
- `contexts/memory/patterns/conventions.md`
- `contexts/memory/project/context.md`
- Relevant codebase files (as identified in execution plan)

---

## Skills

- `implement` — code generation against acceptance criteria
- `frontend-design` — distinctive, production-grade frontend interfaces (activate when execution plan involves UI components, pages, or visual design — see `specialists.registry.yaml` ui-component triggers)
- `codebase-scan` — map affected files and dependencies before implementing
- `context-building` — assemble focused coding context per plan step
- `consistency-check` — validate implementation against plan before handoff

---

## Specialist Dispatch

The Implementation Sub-Agent reads the execution plan and matches against `agents/specialists.registry.yaml`. For each trigger match, it spawns the corresponding Specialist Sub-Agent:

```
Specialist lifecycle: spawn → execute → handoff-specialist-artefact → die
```

Specialist handoff artefacts are written to `contexts/artefacts/impl-reports/{id}.specialist-{domain}.md`.

The Implementation Sub-Agent waits for each specialist handoff before proceeding to the next plan step that depends on it.

---

## Handoff Artefact

Writes to: `contexts/artefacts/impl-reports/{id}.impl-report.md`

The artefact must include:
