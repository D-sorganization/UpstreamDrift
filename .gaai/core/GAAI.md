# GAAI — Master Orientation

Welcome. This is the `.gaai/` folder — the GAAI framework living inside your project.

---

## What Is This Folder?

`.gaai/` contains everything needed to run an AI-assisted SDLC with governance:

```
.gaai/
├── README.md               ← start here (human + AI onboarding)
├── GAAI.md                 ← you are here (full reference)
├── QUICK-REFERENCE.md      ← daily cheat sheet
├── VERSION                 ← framework version
│
├── core/                   ← framework engine (auto-synced to OSS via post-commit hook)
│   ├── agents/             ← who reasons and decides
│   ├── skills/             ← what gets executed
│   ├── contexts/rules/     ← governance (what is allowed)
│   ├── workflows/          ← how the pieces connect
│   ├── scripts/            ← bash utilities
│   └── compat/             ← thin adapters per tool
│
└── project/                ← YOUR project data (never overwritten by updates)
    ├── agents/             ← custom agents (project-specific)
    ├── skills/             ← custom skills (domains/, cross/)
    ├── contexts/
    │   ├── rules/          ← rule overrides
    │   ├── memory/         ← durable knowledge
    │   ├── backlog/        ← execution queue
    │   └── artefacts/      ← evidence and traceability
    ├── workflows/          ← custom workflows
    ├── scripts/            ← custom scripts
    └── content/            ← content drafts
```

**Resolution pattern:** for agents, skills, and rules — the framework loads `core/` first, then `project/` as extension/override.

**This folder contains governance files, not application code.** When scanning the codebase for application logic, there is no need to load `.gaai/` — its files are loaded explicitly by agents when needed, never automatically.

---

## How to Navigate

**If you are adding GAAI to an existing project:**
→ Start with `core/agents/bootstrap.agent.md`. The Bootstrap Agent is your entry point.
→ Its job: scan the codebase, extract architecture decisions, normalize rules, build memory.
→ Run `core/workflows/context-bootstrap.workflow.md` to guide the Bootstrap Agent through initialization.
→ Bootstrap completes when memory, rules, and decisions are all captured and consistent.
→ After bootstrap: switch to Discovery or Delivery depending on your current work.

**If you are just starting a new project:**
→ Read `core/agents/README.agents.md` to understand who does what.
→ Then look at `core/workflows/context-bootstrap.workflow.md` to start your first session.

**If you want to understand the skills:**
→ Read `core/skills/README.skills.md` for the full catalog.
→ Each skill lives in its own directory with a `SKILL.md` file.

**If you want to customize rules:**
→ Add override files in `project/contexts/rules/`. Start with `core/contexts/rules/orchestration.rules.md` as reference.

**If you want to switch to a different AI tool:**
→ Read `core/compat/COMPAT.md` for the compatibility matrix and instructions.
→ Re-run `install.sh --tool <tool> --yes` from the GAAI framework repo. There is no other adapter deployment mechanism.

---

## First Steps

**Existing project (onboarding GAAI onto an existing codebase):**
