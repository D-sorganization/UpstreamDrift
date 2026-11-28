# Integration Testing Strategy

This document defines the intended structure for UpstreamDrift's integration and cross-engine test coverage.

## Goals

- Keep the default PR lane fast and honest.
- Make optional native-engine coverage explicit instead of implied.
- Distinguish missing dependencies from broken expected environments.
- Make it easy to add new engine-specific lanes later.

## Test Tiers

### Tier 1: Core Unit Tests

Location:

- `tests/unit/`

Characteristics:

- Dependency-light by default
- Safe to run in regular PR CI
- May mock native-engine boundaries when the purpose is adapter logic

Examples:

- `tests/unit/test_process_isolation_strict.py`
- `tests/unit/isolated/test_drake_strict.py`
- `tests/unit/isolated/test_pinocchio_strict.py`

### Tier 2: Core Integration Tests

Location:

- `tests/integration/`

Characteristics:

- Exercise real component boundaries and wiring
- May skip if an optional engine is not installed
- Should use real data flow instead of mock-only assertions

Examples:

- `tests/integration/test_real_engine_loading.py`
- `tests/integration/test_engine_integration.py`

### Tier 3: Cross-Engine Validation

Primary files:

- `tests/integration/test_cross_engine_validation.py`
- `tests/fixtures/fixtures_lib.py`

Characteristics:

- Compares behavior across multiple real engines
- Requires at least two ready engines for meaningful execution
- Uses tolerance-based assertions instead of byte-for-byte equality

### Tier 4: Dedicated Native-Engine Lanes

Current direction:

- `.github/workflows/nightly-cross-engine.yml`

Characteristics:

- Expected to provision the engines intentionally
- Must fail if expected native imports are broken
- Should not silently degrade to "all skipped" when the lane exists specifically to validate those engines

## Core CI vs Native-Engine CI

The default `ci-standard.yml` test lane is a **core** lane.

What it means:

- It installs `.[dev,gui-test]`.
- MuJoCo may run because it is part of the base package dependencies.
- Drake, Pinocchio, OpenSim, and MyoSuite are not intentionally provisioned there.

What it does **not** mean:

- A green PR check does not prove full native-engine integration coverage for all optional stacks.

## Engine Probe Semantics

Shared integration fixtures now distinguish three states:

- `missing`: the optional engine dependency is not installed
- `broken`: the dependency is installed but failed to initialize or load the fixture
- `ready`: the dependency loaded successfully for the fixture

This state is carried by `EngineInstance.status` in `tests/fixtures/fixtures_lib.py`.

## Strict Engine Probe Mode

Environment variable:

```bash
UPSTREAM_DRIFT_STRICT_ENGINE_PROBES=true
```

Behavior:

- In regular local or core CI runs, broken optional engines can still degrade to skipped cross-engine coverage.
- In dedicated native-engine lanes, strict mode should be enabled so installed-but-broken engines fail loudly.

This keeps optional local workflows ergonomic while making dedicated validation lanes trustworthy.

## Recommended Expansion Path

When adding more native-engine coverage, prefer separate lanes over growing the core PR job:

1. `test-core`
2. `test-mujoco`
3. `test-drake`
4. `test-pinocchio`
5. `test-opensim`
6. `test-myosuite`
7. `test-cross-engine`

Each dedicated lane should:

- install only the extras it intends to validate
- verify native imports before running tests
- set `UPSTREAM_DRIFT_STRICT_ENGINE_PROBES=true`
- publish artifacts that make failures diagnosable

## Maintainer Notes

If you are unsure where a new test belongs:

- If it mocks native modules to test adapter logic, it belongs in `tests/unit/`.
- If it exercises real component wiring with real imports, it belongs in `tests/integration/`.
- If it compares multiple engines numerically, it belongs in the cross-engine suite.

The goal is not just more tests; it is clearer signal about what passed and why.

## Cross-Engine CI Claim Matrix

This table maps **documented acceptance claims** to the workflow jobs that
execute them. It is authoritative over informal module docstrings when they
disagree (issue #15245, finding F-UD-013).

| Claim | Spec / issue | CI workflow | Job | Primary test module |
| --- | --- | --- | --- | --- |
| Contact-free gravity-only grip agreement (≤ 5 mm among installed engines) | #4249, #7048 | `cross-engine-equivalence.yml` | `equivalence` | `tests/motion_matching/test_cross_engine_equivalence.py` |
| JaxSim ↔ Pinocchio free-body dynamics parity | #6881, #14290 | `cross-engine-equivalence.yml` | `equivalence` | `tests/cross_engine/test_jaxsim_vs_pinocchio.py` |
| Flex-shaft / forward-sim JaxSim hooks | — | `cross-engine-equivalence.yml` | `equivalence` | `tests/cross_engine/test_flex_shaft_parity.py`, `test_jaxsim_forward_sim.py` |
| Impact-parameter adapter conformance | — | `cross-engine-equivalence.yml` | `equivalence` | `tests/unit/impact_parameters/test_adapters.py` |
| Canonical Core adapter validation | CC-8 | `cross-engine-equivalence.yml` | `canonical-conformance-gate` | `tests/integration/test_cross_engine_validation.py`, `tests/integration/cross_engine/**` |
| Per-engine force-overlay parity | #11346 | `cross-engine-equivalence.yml` | `force-overlay-parity` | `tests/integration/cross_engine/test_force_overlay_parity.py` |
| Coefficient / metric helper smoke (no CI gate) | #4096 helpers | _not scheduled_ | — | `tests/cross_engine/test_cross_engine_equivalence_helpers.py` |
| Actuated `theta_truth` §2.2 forward-sim 5 mm gate (full six-engine campaign) | `CROSS_ENGINE_PARITY_SPEC.md` §2.2 | _not authorized in #15245_ | — | _planned / epic #13789_ |

Helper tests under `tests/cross_engine/test_cross_engine_equivalence_helpers.py`
must not be mistaken for the gravity-only gate. The workflow does not execute
them today.
