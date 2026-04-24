# Specification Workflow

This directory is the home for maintained development specifications for this repository.

Use a spec when work changes architecture, workflows, APIs, schemas, data contracts, security boundaries, deployment behavior, or spans multiple files in a way that should remain traceable after merge. Trivial typo fixes, dependency bumps, and isolated non-behavioral cleanup do not need a spec.

## Layout

- `active/` - approved or in-progress specs guiding current implementation.
- `planned/` - proposed specs that are not yet approved for implementation.
- `implemented/` - specs retained for recently shipped behavior.
- `../archive/specs/` - superseded or retired specs that should remain discoverable.

## Minimum Spec Content

Each active spec should include:

- status: `proposed`, `active`, `implemented`, `superseded`, or `archived`
- problem statement
- scope
- non-goals
- architecture or design notes
- acceptance criteria
- validation or test expectations
- links to governing issues, PRs, and implementation branches

## Implementation Rules

- Reference the governing spec from substantial implementation PRs.
- Update the spec when implementation changes behavior, scope, acceptance criteria, or validation expectations.
- Move superseded specs to `docs/archive/specs/` and add an index entry when they are still useful for historical context.
- Prefer compact, maintained specs over broad documents that become stale.
