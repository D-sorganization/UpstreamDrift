# Unified Agent Governance Directory Map

## Purpose and Authority

This directory serves as the unified navigation index for agent configurations, tool registrations, and execution workflows across `UpstreamDrift`. Multiple agent frameworks interface with this repository; each framework maintains dedicated metadata and operational assets under its respective top-level dot-directory.

The single sources of truth for repository policies and shared infrastructure remain:

- [`CLAUDE.md`](../CLAUDE.md): Repository coding standards, test gates, formatting rules, verification commands, and file size budgets.
- [`AGENTS.md`](../AGENTS.md): Shared infrastructure directory, modeling references, coordinate conventions, and discovery workflows.

---

## Agent Framework Directory Map

### `.agent` — Agent Skills and Workflows

Contains cross-agent reusable skills and composite workflow declarations:

- `.agent/skills/`: Executable skill procedures and definitions.
- `.agent/workflows/`: Multi-step orchestration routines.

### `.claude` — Claude Configuration and Skills

Houses Claude-specific workspace commands, prompt augmentations, and tool skills:

- `.claude/commands/`: Custom command extensions and task scripts.
- `.claude/skills/`: Specialized domain skills for Claude agents.

### `.gaai` — GAAI Project and Automation Scripts

Stores Global Artificial Autonomous Intelligence (GAAI) workspace configurations:

- `.gaai/core/`: Engine bindings and core agent protocols.
- `.gaai/project/`: Repository-specific project context and metadata.
- `.gaai/scripts/`: Automation scripts and execution hooks.

### `.gemini` — Gemini Command Specifications

Contains Gemini-targeted command manifests and agent protocol templates:

- `.gemini/commands/`: Subcommand declarations and prompt blueprints.

### `.jules` — Jules Autonomous Agent Policies and Templates

Manages Jules agent operational policies, task tracking, and guardrails:

- `.jules/bolt.md`: Fast-path execution rules and safety bounds.
- `.jules/sentinel.md`: Continuous verification and regression sentinels.
- `.jules/pending/`: Queued agent run manifests.
- `.jules/templates/`: Reusable task packet and PR description templates.

### `.kiro` — Kiro Launcher and Steering Directives

Maintains Kiro-specific runner configurations:

- `.kiro/launcher/`: Agent execution launcher scripts.
- `.kiro/steering/`: Behavioral steerage and context injection files.

### `.specify` — Spec-Driven Workflows and Integrations

Encapsulates specification-driven engineering workflows:

- `.specify/extensions/`: Spec-kit extensions and configuration schemas.
- `.specify/memory/`: Spec state and history tracking.
- `.specify/templates/`: Template specifications and test contracts.
- `.specify/workflows/`: Automated lifecycle workflows.

---

## Governance Rules for AI Agents

1. **Policy Hierarchy:** Project policy in `CLAUDE.md` and scientific/infrastructure standards in `AGENTS.md` supersede any tool-specific recommendation.
2. **Worktree Isolation:** External agents must operate strictly in isolated external worktrees outside the repository root.
3. **Root Clutter Law:** No agent may write untracked files or directories directly to the repository root. All agent state files must reside in designated `.ai` or framework-specific directories.
4. **Verification Gates:** All changes must satisfy local pre-commit checks, type checks, and verification suites prior to branch submission.
