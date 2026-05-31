# Jules Task Template: Canonical Core Docstring Pass

Use this template when tightening adapter documentation after implementation.

## Branch and PR

- Base branch: `main`
- Branch prefix: `docs/canonical-core-<engine>-docstrings`
- PR state: draft if it changes executable examples or any adapter code
- PR body must include: `Closes #<issue>` or `Refs #<issue>` and a conformance
  gate note.

## Scope

- Document public adapter methods with preconditions, postconditions, units, and
  canonical-v2 layout assumptions.
- Link to `docs/conventions/canonical-v2.md` instead of restating the full
  state contract.
- Do not change runtime behavior in a documentation-only pass.
- If examples become executable, keep them light and exclude heavy engine
  imports unless the matching `requires_*` marker is used.

## Required Checks

- `python3 -m ruff check <changed paths>`
- `python3 -m ruff format --check <changed paths>`
- `python3 -m pytest tests/unit/pose_interchange/adapters -q`
- The PR must not be marked ready until `Canonical Core Conformance Gate` passes
  when adapter-facing docs or examples changed.
