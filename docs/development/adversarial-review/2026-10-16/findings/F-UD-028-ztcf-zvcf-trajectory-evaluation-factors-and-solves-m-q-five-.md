# F-UD-028

**ZTCF/ZVCF trajectory evaluation factors and solves `M(q)` five times per sample**

| Severity | Category    | Confidence | Effort | Status   |
| -------- | ----------- | ---------- | ------ | -------- |
| ⬜ Low   | performance | high       | S      | proposed |

**Locations:** `src/shared/python/simulation_backends/ztcf_zvcf.py:553`, `src/shared/python/simulation_backends/ztcf_zvcf.py:498`

## Finding

- Severity: Low - Category: performance
- Confidence: high
- Location: src/shared/python/simulation_backends/ztcf_zvcf.py:553-558; src/shared/python/simulation_backends/ztcf_zvcf.py:498-512
- Evidence: per sample: `ztcf_acceleration` (M + solve), `zvcf_acceleration` (M + solve), `zero_velocity_control_preserved_
acceleration` (M + solve), `drift_and_control_split` → `ztcf_acceleration` again + `_solve_mass` (M + solve ×2).
- Failure scenario: for a 25-DoF humanoid at 1 kHz over 2 s that is 10,000 extra Cholesky factorisations and 2,000 redundant
  `mass_matrix` calls through the engine binding (the dominant cost for Pinocchio/MuJoCo providers); `drift` is guaranteed equal
  to `ztcf` by the postcondition and is computed twice anyway.
- Why a newer engineer misses this: composing small pure functions is good style; the shared sub-expression is hidden behind
  the provider protocol.
- Proposed change: compute `M = provider.mass_matrix(q)`, `cho = cho_factor(M)`, `h_v = bias(q, v)`, `h_0 = bias(q, 0)` once and
  derive all five vectors with `cho_solve`.
- Verification: `provider.mass_matrix` call count == `num_steps` in a counting-mock test; outputs bit-identical.
- Effort: S

## Draft Pull Request

**Title:** `perf(ztcf): factor M(q) once per sample and reuse across all five solves`

### Why This Matters (The Pitch)

Evaluating a ZTCF/ZVCF trajectory builds and solves the mass matrix five times per sample. For a 25-DoF humanoid at 1 kHz over 2 s that is 10,000 extra Cholesky factorisations and 2,000 redundant mass_matrix calls through the engine binding, the dominant cost for Pinocchio and MuJoCo providers, and the drift term is computed twice though it is guaranteed equal to ztcf. Computing M, its Cholesky factor and the two bias vectors once and deriving everything with cho_solve cuts this cost with bit-identical output.

### Proposed PR Body

## Problem

`ztcf_zvcf.py:553-558` and `:498-512`: per sample, `ztcf_acceleration`, `zvcf_acceleration`, `zero_velocity_control_preserved_acceleration` and `drift_and_control_split` (which calls `ztcf_acceleration` again plus `_solve_mass`) each rebuild `M` and solve.

## Change

- [ ] Compute `M = provider.mass_matrix(q)`, `cho = cho_factor(M)`, `h_v = bias(q, v)`, `h_0 = bias(q, 0)` once
- [ ] Derive all five vectors with `cho_solve`

## Verification

`provider.mass_matrix` call count == `num_steps` in a counting-mock test; outputs bit-identical.

## Risk and Rollback

Low: pure refactor guarded by an identical-output test. Rollback is a revert.

## Handoff

None

Refs: adversarial review 2026-10-16, F-UD-028

---

_Generated 2026-10-16 from an adversarial review of `UpstreamDrift` @ `87d6d6ee`. This is a proposal, not a verdict — verify the cited lines before acting._
