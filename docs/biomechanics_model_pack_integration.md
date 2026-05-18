# Biomechanics Model Pack Integration Plan

**Parent Issues:**

- [#5312](https://github.com/D-sorganization/UpstreamDrift/issues/5312) - Incorporate MuJoCo/Drake/Pinocchio/OpenSim model packs
- [#5313](https://github.com/D-sorganization/UpstreamDrift/issues/5313) - Reconcile model_pack/v1 provider schema
- [#5314](https://github.com/D-sorganization/UpstreamDrift/issues/5314) - Expose and categorize all runnable tools

**Parent EPIC:** [#5309](https://github.com/D-sorganization/UpstreamDrift/issues/5309)
**Date Created:** 2026-05-12
**Status:** Planning

## Overview

This document addresses three interconnected issues related to biomechanics model packs and tool exposure in UpstreamDrift:

1. **Issue #5312**: Incorporate four biomechanics model packs (MuJoCo, Drake, Pinocchio, OpenSim)
2. **Issue #5313**: Reconcile model_pack/v1 provider schema with UpstreamDrift manifest contract
3. **Issue #5314**: Expose and categorize all runnable UpstreamDrift tools

## Current State

### Live Model Packs

All four biomechanics provider repos have `model_pack.yaml` on `origin/main`:

| Repo             | Manifest | Model Pack Module                    | Tests |
| ---------------- | -------- | ------------------------------------ | ----- |
| MuJoCo_Models    | ✅       | `src/mujoco_models/model_pack.py`    | ✅    |
| Drake_Models     | ✅       | `src/drake_models/model_pack.py`     | ✅    |
| Pinocchio_Models | ✅       | `src/pinocchio_models/model_pack.py` | ✅    |
| OpenSim_Models   | ✅       | `src/opensim_models/model_pack.py`   | ✅    |

### Schema Mismatch

The external model repos use a simple `model_pack/v1` manifest, while UpstreamDrift's stricter loader expects:

- `manifest_version`
- `pack_id`
- `pack_name`
- `provider`
- `models` array with specific fields

## Integration Strategy

### Phase 1: Schema Reconciliation (Issue #5313)

**Goal**: Create an adapter that normalizes `model_pack/v1` manifests into UpstreamDrift's contract.

#### Implementation Steps

1. **Create Schema Adapter** (`src/shared/python/biomech/model_pack_adapter.py`):

   ```python
   class ModelPackAdapter:
       """Normalizes model_pack/v1 manifests into UpstreamDrift contract."""

       def normalize(self, manifest: dict) -> dict:
           """Convert model_pack/v1 to UpstreamDrift format."""

       def validate(self, manifest: dict) -> tuple[bool, str]:
           """Validate manifest and return actionable errors."""
   ```

2. **Add Fixture Tests** (`tests/test_model_pack_normalization.py`):

   - Test valid `model_pack/v1` manifests from all four providers
   - Test malformed manifests produce actionable errors
   - Test normalized output includes required metadata

3. **Update Documentation**:
   - `docs/model_pack_spec.md` - Updated contract specification
   - Provider repos: Update `README.md` with normalization notes

#### Acceptance Criteria

- [ ] Current provider manifests load without manual rewriting
- [ ] Malformed manifests fail with actionable messages
- [ ] Normalized output includes: category, provider, exercises, engine, display metadata

### Phase 2: Model Pack Registration (Issue #5312)

**Goal**: Register all four biomechanics providers under Biomechanics category.

#### Implementation Steps

1. **Create Provider Registry** (`src/shared/python/biomech/provider_registry.py`):

   ```python
   BIOMECHANICS_PROVIDERS = {
       "mujoco": {
           "repo": "MuJoCo_Models",
           "display_name": "MuJoCo Biomechanics",
           "exercises": ["squat", "deadlift", "bench_press", ...],
       },
       "drake": {...},
       "pinocchio": {...},
       "opensim": {...},
   }
   ```

2. **Update Launcher** to consume provider registry:

   - Add Biomechanics category
   - Register exercises under appropriate providers
   - Add visibility tests

3. **Add Discovery Tests** (`tests/test_biomechanics_discovery.py`):
   - All four providers visible under Biomechanics
   - At least one exercise entry per provider
   - Missing repo handling
   - Malformed manifest handling

#### Acceptance Criteria

- [ ] All four providers visible under Biomechanics
- [ ] Tests cover discovery, category, and exercise entries
- [ ] Docs explain sharing/versioning for future providers

### Phase 3: Tool Exposure (Issue #5314)

**Goal**: Inventory and categorize all runnable UpstreamDrift tools.

#### Categories

| Category           | Tools                                        |
| ------------------ | -------------------------------------------- |
| Physics Engines    | Drake, MuJoCo, Pinocchio, OpenSim simulators |
| Biomechanics       | Model pack viewers, exercise analyzers       |
| Simulation         | Forward dynamics, inverse kinematics         |
| Motion Matching    | Database search, trajectory extraction       |
| Motion Capture     | C3D processing, marker tracking              |
| Analysis           | Swing analysis, joint angle extraction       |
| Documentation      | Spec viewers, ADR browser                    |
| External Providers | Ollama, OpenAI adapters                      |
| Developer Tools    | Test runners, schema validators              |

#### Implementation Steps

1. **Create Tool Inventory** (`src/launchers/tool_inventory.py`):

   - Scan source for runnable tools
   - Cross-check with launcher manifests
   - Identify hidden/missing entries

2. **Update Launcher Categories**:

   - Add missing categories
   - Fix `hidden: true` entries with documentation
   - Add coverage tests

3. **Add Coverage Tests** (`tests/test_launcher_coverage.py`):
   - Inventory vs launcher comparison
   - Hidden entries must have documented reason

#### Acceptance Criteria

- [ ] All supported tools appear in correct categories
- [ ] Hidden features are manifest-listed with reason and owner
- [ ] Feature-discovery tests prevent regressions

## TDD / DbC / LOD / DRY Principles

- **TDD**: Every phase adds focused regression tests
- **DbC**: Schema validation produces actionable errors
- **LOD**: Providers consume public adapter APIs
- **DRY**: One adapter path handles all providers

## PR Tracking

| Phase                      | Issue | PR # | Status     |
| -------------------------- | ----- | ---- | ---------- |
| Phase 1: Schema Adapter    | #5313 | -    | ⏳ Pending |
| Phase 2: Provider Registry | #5312 | -    | ⏳ Pending |
| Phase 3: Tool Exposure     | #5314 | -    | ⏳ Pending |

## Related Issues

- #5307 - Ollama path deduplication (fixed via PR #5326)
- #5316 - End-to-end smoke tests (PR #5327)
- #5315 - Chat UI recovery plan (PR #5333)

---

_This document will be updated as integration PRs are created and merged._
