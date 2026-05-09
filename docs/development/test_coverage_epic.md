# Epic: Phase 3 - Systematic Coverage Ratcheting and Untested Module Remediation

## Executive Summary

Following the successful completion of Phase 2 (Test Suite Modularization), where oversized test files (>500 lines) were systematically decomposed and legacy dead tests were archived, the UpstreamDrift repository has achieved a stabilized baseline. However, global test coverage remains stagnant around the ~55% threshold due to a significant backlog of entirely untested core modules.

This Epic initiates **Phase 3: Coverage Ratcheting**, with the ultimate goal of achieving the organizational standard of 80% global coverage. This will be accomplished through systematic identification of untested modules, contract-based integration testing, and a strict CI/CD ratchet.

## Objectives

1. **Identify and Classify Untested Modules:** Audit the entire `src/` tree to catalog the 800+ modules lacking test coverage, categorizing them by architectural tier (e.g., Domain Core vs. UI Launchers).
2. **Prioritize High-Risk Domains:** Focus initial testing efforts on critical physics engines, API contracts, and spatial algebra modules that represent the highest technical risk.
3. **Generate High-Quality Contract Tests:** Implement rigorous, decoupled tests utilizing `pytest` and `unittest.mock` to validate module boundaries and error-handling paths.
4. **Enforce Monotone Coverage Growth:** Leverage the newly deployed `scripts/config/coverage_enforcer.py` in CI/CD to prevent regressions, systematically raising the required coverage floor package-by-package.

## Implementation Plan

### Sprint 1: Global Coverage Audit and Baseline Generation

- **Action:** Run a comprehensive `pytest --cov=src` pass and export the `coverage.xml` report.
- **Action:** Parse the coverage report to generate a canonical "Hit List" of modules with 0% coverage.
- **Deliverable:** A prioritized dashboard artifact categorizing untested modules by severity and impact.

### Sprint 2: Core Domain Remediation (Physics & Spatial Algebra)

- **Action:** Target `src/shared/python/spatial_algebra/` and `src/shared/python/physics_core/` for immediate remediation.
- **Action:** Implement parameterized tests covering boundary conditions, singular matrix handling, and cross-engine protocol compliance.
- **Deliverable:** Core math domains elevated to >85% coverage.

### Sprint 3: API and Backend Services Remediation

- **Action:** Target `src/api/routers/` and `src/api/services/`.
- **Action:** Implement `httpx` and `AsyncMock` based integration tests for all REST and WebSocket endpoints, focusing on authentication flows and payload validation.
- **Deliverable:** API domain elevated to >80% coverage.

### Sprint 4: CI/CD Ratchet Enforcement

- **Action:** Update `pyproject.toml` and CI pipeline configurations to harden the per-package coverage thresholds based on the gains achieved in Sprints 1-3.
- **Deliverable:** The global baseline permanently locked at a higher threshold, preventing future PRs from introducing untested code.

## Acceptance Criteria

- The global repository coverage metric reaches or exceeds **80%**.
- No individual core module (`src/shared/`, `src/api/`, `src/engines/`) falls below 75% coverage.
- All new tests adhere to the established "Independent Class" paradigm, ensuring CI parallelization is uninhibited.
- The CI/CD coverage enforcer successfully gates PRs that reduce coverage.
