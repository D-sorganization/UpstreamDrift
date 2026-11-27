# Agent Configuration Ownership

Issue #3835 identified five root-level agent configuration trees. This file is
the ownership inventory and migration ledger. It does not authorize deletion by
itself; deletion requires a follow-up PR that proves tool references have moved
and that the corresponding workflow owners approve the migration.

`CLAUDE.md` and `AGENTS.md` are the binding contributor and agent policy. The
vendored framework tree that used to be the canonical root was retired
fleet-wide and removed (Repository_Management #2725); `.claude/` is now the
canonical agent configuration root.

| Directory | Status    | Owner   | Migration note                                                                                                          |
| --------- | --------- | ------- | ----------------------------------------------------------------------------------------------------------------------- |
| .claude/  | Canonical | @core   | Claude skills, rules, and settings.                                                                                     |
| .kiro/    | Legacy    | @agents | Tool-specific steering/config root; migrate unique steering guidance into `AGENTS.md` before deletion.                  |
| .jules/   | Legacy    | @agents | Jules-specific automation configuration; migrate unique ownership metadata into `.github/WORKFLOWS.md` before deletion. |

## Ownership Rules

- New agent governance belongs in `AGENTS.md` or its fleet-rules sources unless
  a tool requires a configuration tree of its own.
- Every root-level agent configuration tree must have an owner and migration
  note in this file.
- Workflow automation owned by agents must have a row in `.github/WORKFLOWS.md`
  with an explicit owner and `KEEP`, `MERGE`, `DELETE`, or `DISABLE` in the
  purpose cell.
- Removing `.kiro/` or `.jules/` is intentionally left to a smaller follow-up
  after owners verify tool references.
