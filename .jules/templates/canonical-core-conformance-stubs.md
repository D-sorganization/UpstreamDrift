# Jules Task Template: Canonical Core Conformance Tests

Use this template when adding the first conformance coverage for an adapter or
extending the CC-7 harness.

## Branch and PR

- Base branch: `main`
- Branch prefix: `feat/canonical-core-<engine>-conformance`
- PR state: draft until the conformance and engine-marker lanes are green
- PR body must include: `Closes #<issue>` and list any expected divergence
  registry entries.

## Scope

- Add tests under `tests/integration/cross_engine/` for CC-7 harness coverage.
- Keep engine-heavy tests marked with the existing marker for that engine:
  `requires_pinocchio`, `requires_drake`, `requires_jaxsim`, or
  `requires_mujoco`.
- Keep CPU-only conformance separate from `live_simulation` so light PRs are not
  blocked by unavailable heavy engines.
- Register justified cross-engine divergence in
  `tests/integration/cross_engine/divergence_registry.yaml`; unregistered
  divergence must fail.

## Required Checks

- `python3 -m pytest tests/integration/test_cross_engine_validation.py tests/integration/cross_engine -q`
- `python3 -m pytest -m "not requires_pinocchio and not requires_drake and not requires_jaxsim and not requires_mujoco" tests/unit/pose_interchange/adapters -q`
- The PR must not be marked ready until `Canonical Core Conformance Gate` passes.
