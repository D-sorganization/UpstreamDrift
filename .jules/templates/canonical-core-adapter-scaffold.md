# Jules Task Template: Canonical Core Adapter Scaffold

Use this template when creating or updating an engine adapter for the Canonical
Core epic.

## Branch and PR

- Base branch: `main`
- Branch prefix: `feat/canonical-core-<engine>-adapter`
- PR state: draft until the conformance lane is green
- PR body must include: `Closes #<issue>` and a note that
  `.github/workflows/cross-engine-equivalence.yml` runs
  `Canonical Core Conformance Gate`

## Scope

- Extend the existing adapter module under `src/shared/python/pose_interchange`
  or the engine-owned package under `src/engines`.
- Do not create a parallel adapter registry.
- Cite `docs/conventions/canonical-v2.md` and `docs/adr/0026-canonical-dynamic-state-v2.md`
  when implementing state layout, quaternion order, or velocity order.
- Keep optional engine imports guarded so standard CI can import the package
  without the heavy engine wheel installed.

## Required Checks

- `python3 -m ruff check <changed paths>`
- `python3 -m ruff format --check <changed paths>`
- `python3 -m pytest tests/unit/pose_interchange/adapters -q`
- The PR must not be marked ready until `Canonical Core Conformance Gate` passes.
