# CI Optional Stack Runbook (#14227)

This note records what is **mechanically fixed on `main`** versus what still
needs an **owner decision** in the scheduled
[CI Optional Stack](https://github.com/D-sorganization/UpstreamDrift/blob/main/.github/workflows/ci-optional-stack.yml)
lane.

## Slice 1 — Pinocchio Gravity Matches the Shared Constant (mechanical, on main)

Issue [#13799](https://github.com/D-sorganization/UpstreamDrift/pull/13799) moved
the CasADi swing model to the repository gravity constant
`GRAVITY_M_S2` (`src/shared/python/core/physics_constants.py`). Pinocchio's
builder previously kept the library default (9.81 m/s²), so symbolic RNEA checks
and `pin.rnea` disagreed in Optional Stack trajectory-optimization steps.

**Fix (merged; completes Pinocchio's side of #13799):**

- `build_pinocchio_model` in `src/shared/python/optimization/model_provider.py`
  sets `model.gravity.linear = [0, 0, -GRAVITY_M_S2]` after
  `pin.buildModelFromXML(...)`.

**Regression lock:**

- `test_pinocchio_model_uses_the_shared_gravity_constant` in
  `tests/unit/optimization/test_model_provider.py` (hermetic mock of pinocchio).

CI contract tests in `tests/ci/test_optional_stack_gravity_contract_14227.py`
assert that this runbook, the builder, the constant, and the regression test
stay wired.

## Slice 2 — Manufactured Authority Regeneration (owner-blocked)

Optional Stack jobs `articulated-manufactured-authority` and
`articulated-manufactured-rolling` still fail when the publication authority
committed on `main` does not match a deterministic regeneration from current
model provenance (first regression around #13039 / #13839). **Agents must not
fabricate or silently re-commit authority artifacts** to turn CI green.

**Owner action required:**

1. Regenerate the manufactured publication authority from `main` using the
   pinned authority environment documented in issue #14227.
2. Re-run rolling conformance once the committed authority matches that
   regeneration.

Until the owner lands that regeneration, Slice 2 remains red and is **not** a
mechanical lane fix.

## Related — Multiple-Shooting Dynamics Premise (#14227 item 3)

`test_multiple_shooting_satisfies_its_own_dynamics` may still fail because the
reference swing is no longer "intentionally unresolved" after the gravity
constant alignment. That threshold has **not** been widened; the owner must
re-pick the reference case or the test premise.
