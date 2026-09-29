# Testing Improvement Action Plan

Date: 2026-03-15
Repository: UpstreamDrift
Status: Planned

## Objective

Make `UpstreamDrift` verify the real shared-module boundary with `Tools`, especially for Python packages and vendored/shared paths, while preserving strong Rust integration.

## Key Problems

1. Shared Python path ordering prefers local copies before vendored `Tools`, which weakens consumer validation.
2. Dedicated `Tools` integration tests are too optional and too tolerant of failure.
3. The suite contains substantial skip/xfail debt in high-value integration areas.
4. Rust integration is materially better enforced than Python shared-module integration.

## Desired End State

1. Required CI proves the vendored or sibling `Tools` packages work as consumed by `UpstreamDrift`.
2. Shared Python tests can distinguish local-copy coverage from downstream-consumer coverage.
3. Cross-repo integration tests fail hard when shared contracts break.
4. Optional engine stacks remain isolated, but core shared-module coverage stays required and deterministic.

## Workstreams

### 1. Separate Local Shared-Code Tests from Consumer Tests

- Introduce explicit test modes:
  - local shared-module tests
  - vendored/sibling `Tools` consumer-contract tests
- Remove ambiguity caused by Python path precedence.

### 2. Strengthen Tools Integration Tests

- Replace always-pass or documentary assertions with real contract checks.
- Remove permissive patterns such as optional import checks that only assert a boolean.
- Convert core `model_generation`, `signal_toolkit`, and shared-theme integration checks into required tests.

### 3. CI and Path Hardening

- Add a required shared-tools consumer-contract job.
- Validate the effective source of imported modules during tests.
- Keep local-package tests and vendored-consumer tests separate in reporting.

### 4. Skip/XFail Debt Reduction

- Audit skip-heavy integration areas and classify them into:
  - must be required
  - optional but quarantined
  - obsolete and removable
- Focus first on shared-module and cross-engine validator paths.

## Verification Criteria

1. Required CI proves `UpstreamDrift` can consume the real shared `Tools` packages.
2. Shared-module imports report and verify the actual provider path.
3. The dedicated `Tools` integration suite fails when real consumer behavior breaks.
4. Rust and Python cross-repo integration both remain enforced.

## GitHub Tracking

- Meta: `#1887` Testing program: make Tools integration deterministic and required
- `#1888` Separate local shared-module tests from vendored Tools consumer tests
- `#1889` Reduce skip/xfail debt in required shared-module and cross-engine validation suites
- `#1890` Strengthen Tools integration suite and remove permissive always-pass patterns
