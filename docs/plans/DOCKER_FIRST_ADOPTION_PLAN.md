# UpstreamDrift Docker-First Adoption Plan

## Objective

Standardize UpstreamDrift around a Docker-first workflow with a single image family name (`upstream-drift`) and explicit tags by purpose.

## Canonical Image Contract

- `upstream-drift:engine`
  Purpose: launcher-driven physics engine execution (MuJoCo/Drake/Pinocchio workflows)
  Build context: `src/engines/physics_engines/mujoco/`

- `upstream-drift:runtime`
  Purpose: API/backend runtime used by `docker compose`
  Build context: repo root `Dockerfile` (`target: runtime`)

- `upstream-drift:dev`
  Purpose: local test/dev container helper workflows
  Build context: `src/engines/physics_engines/mujoco/Dockerfile` (or explicit dev Dockerfile in future)

## Why Tags Instead of Multiple Names

Using a single repository name with tags keeps identity consistent while avoiding local clobbering between different Docker build contexts.

## Current Migration Scope (in progress)

- Rename host conda env to `upstream-drift`
- Move launcher Docker image references to `upstream-drift:engine`
- Move compose image reference to `upstream-drift:runtime`
- Add compatibility fallback for legacy local tags (`robotics_env`, `golf-suite`)
- Update tests/scripts/docs to canonical tags

## Rollout Phases

1. Phase 1: Compatibility Layer (current)

- Keep legacy fallback while emitting warnings.
- Add retag guidance and migration docs.

2. Phase 2: Default Docker-First UX

- Launcher defaults to Docker mode when available.
- Local/WSL remain explicit fallback modes.

3. Phase 3: Runtime Tiering

- Keep core product runtime lean.
- Move heavy training dependencies to optional training profile.

4. Phase 4: Removal of Legacy Names

- Remove fallback aliases after one release cycle and migration completion.

## CI/Quality Gates (required)

- Build and smoke-test `upstream-drift:runtime` in CI.
- Build and smoke-test `upstream-drift:engine` in CI.
- Add image-size budget checks with gradual enforcement.

## Reversibility

- Retag commands support rollback at any time:
  - `docker tag upstream-drift:engine robotics_env:latest`
  - `docker tag upstream-drift:runtime golf-suite:latest`
- No destructive migration step should be mandatory.

## Related Issues

- UpstreamDrift: #1555, #1556, #1557, #1558, #1559, #1560, #1561, #1562
- MLProjects: #160, #161
