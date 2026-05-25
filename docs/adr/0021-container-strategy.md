# ADR-0021: Container Strategy

Status: Accepted

## Context

Issue #6097 requires consolidating our container strategy.

## Decision

- `Dockerfile` stays the canonical default release/runtime/training image
- `Dockerfile.heavy_test` stays dedicated to heavy-test parity
- `Dockerfile.modular` stays the supported opt-in profile build surface
