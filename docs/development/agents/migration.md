# Agent Configuration Ownership

Issue #3835 identified five root-level agent configuration trees. This file is
the ownership inventory and migration ledger. It does not authorize deletion by
itself; deletion requires a follow-up PR that proves tool references have moved
and that the corresponding workflow owners approve the migration.

`.gaai/` is the canonical agent-governance root because `CLAUDE.md` declares the
repository a GAAI Fleet member and `AGENTS.md` delegates binding contributor
policy to `CLAUDE.md`.

| Directory | Status    | Owner   | Migration note                                                                                                                                     |
| --------- | --------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| .gaai/    | Canonical | @core   | Owns governance rules, memory, backlog, skills, hooks, and orchestration docs.                                                                     |
| .claude/  | Mirror    | @core   | Contains Claude command shims and duplicated skill entries; migrate durable content into `.gaai/core/compat/` or `.gaai/project/` before deletion. |
| .agent/   | Legacy    | @agents | Contains older skill and workflow shims; compare with `.gaai/core/skills/` before removing duplicates.                                             |
| .kiro/    | Legacy    | @agents | Tool-specific steering/config root; migrate unique steering guidance into `.gaai/project/contexts/rules/` before deletion.                         |
| .jules/   | Legacy    | @agents | Jules-specific automation configuration; migrate unique ownership metadata into `.gaai/project/` and `.github/WORKFLOWS.md` before deletion.       |

## Ownership Rules

- New agent governance belongs under `.gaai/project/` unless a tool requires a
  compatibility shim elsewhere.
- Compatibility shims outside `.gaai/` must have an owner and migration note in
  this file.
- Workflow automation owned by agents must have a row in `.github/WORKFLOWS.md`
  with an explicit owner and `KEEP`, `MERGE`, `DELETE`, or `DISABLE` in the
  purpose cell.
- Removing `.claude/`, `.agent/`, `.kiro/`, or `.jules/` is intentionally left to
  a smaller follow-up after owners verify tool references.
