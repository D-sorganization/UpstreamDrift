# F-UD-012

**"Numeric parity" Rust test file tests `clamp`, `lerp` and `is not None`**

| Severity  | Category | Confidence | Effort | Status   |
| --------- | -------- | ---------- | ------ | -------- |
| 🟨 Medium | tests    | high       | S      | proposed |

**Locations:** `tests/rust_bindings/test_rust_python_numeric_parity.py:128`, `tests/rust_bindings/test_rust_python_numeric_parity.py:147`, `src/shared/python/physics/rust_kernel.py:10`

## Finding

- Severity: Medium - Category: tests
- Confidence: high
- Location: tests/rust_bindings/test_rust_python_numeric_parity.py:128-139; tests/rust_bindings/test_rust_python_numeric_parity.py:147-158;
  src/shared/python/physics/rust_kernel.py:10-14
- Evidence:
  ```python
  class TestRK4NumericParity:
      """Verify RK4 integrator matches analytical solutions to high precision."""
      def test_config_creation(self) -> None:
          config = upstream_physics.IntegratorConfig(dt=0.01, max_steps=500)
          assert config is not None
  ```
  `rust_kernel.py` claims "All physics calculations route through the same Rust binary" and "TDD: see
  tests/rust_bindings/test_physics_bindings.py"; the facade only constructs config objects.
- Failure scenario: an RK4 regression in `rk4.rs` is invisible to the file whose docstring says it checks RK4 against analytical
  solutions. (The real parity lives in `test_aerodynamics_engine_parity.py`, which is good; the misnamed file dilutes trust.)
- Why a newer engineer misses this: the file was scaffolded from an earlier Tools repo ("mixing entropy lowers G" in its
  docstring is chemistry, not golf) and the class names were kept.
- Proposed change: delete `TestRK4NumericParity`/`TestContactParity` or make them real (projectile with no drag vs closed form,
  `|x(t) − x_exact| < C·dt⁴`).
- Verification: the test fails when `rk4.rs` uses Euler.
- Effort: S

## Draft Pull Request

**Title:** `test(rust): make RK4 parity tests real or delete the is-not-None ones`

### Why This Matters (The Pitch)

A file called test_rust_python_numeric_parity.py has a class TestRK4NumericParity whose docstring promises 'RK4 integrator matches analytical solutions to high precision' and whose test builds an IntegratorConfig and asserts it is not None. rust_kernel.py claims all physics routes through the Rust binary and points to tests that only construct config objects. An RK4 regression in rk4.rs would be invisible to the file whose docstring says it checks RK4. The real parity coverage lives in test_aerodynamics_engine_parity.py and is good, which makes the misnamed file worse: it dilutes trust in the suite. Either delete TestRK4NumericParity and TestContactParity, or make them real with a no-drag projectile against the closed form and an error bound of C times dt to the fourth. Size S, and the test file stops lying about what it covers.

### Proposed PR Body

## Problem

`tests/rust_bindings/test_rust_python_numeric_parity.py:128-139` and `:147-158` define `TestRK4NumericParity` (docstring: "Verify RK4 integrator matches analytical solutions") whose tests only do `config = upstream_physics.IntegratorConfig(dt=0.01, max_steps=500); assert config is not None`. `src/shared/python/physics/rust_kernel.py:10-14` cites this as backing for "All physics calculations route through the same Rust binary".

## Change

- [ ] Delete `TestRK4NumericParity` and `TestContactParity`, or make them real: projectile with no drag vs closed form, `|x(t) − x_exact| < C·dt⁴`

## Verification

The test fails when `rk4.rs` uses Euler.

## Risk and Rollback

Low: test-only change. Rollback is a revert.

## Handoff

- **Branch, commit, and pull request:** `<branch>` @ `<commit>` — PR #<n> (fill in when opened)
- **Validation:** the command(s) under _Verification_ above, run locally and green; plus the repo's fast gates
  (`ruff check`, `ruff format --check`, the relevant `-m unit` subset)
- **Blockers:** none known at drafting time — record any that appear
- **Next step:** open the PR against `main`, arm auto-merge with `scripts/automerge_guard.py`, end the session
- **Follow-ups:** none

Refs: adversarial review 2026-10-16, F-UD-012

---

_Generated 2026-10-16 from an adversarial review of `UpstreamDrift` @ `87d6d6ee`. This is a proposal, not a verdict — verify the cited lines before acting._
