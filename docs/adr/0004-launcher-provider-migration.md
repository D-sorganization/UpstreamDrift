# ADR 0004: Launcher Provider Migration Modes and Legacy Deprecation Policy

- Status: Accepted
- Date: 2026-04-08
- Related issues: #2395, #2397, #2399, #2401, #2404

## Context

UpstreamDrift is migrating from a launcher model that assumes local repository
paths such as `REPOS_ROOT/src/engines/...` to a provider-backed model-pack
architecture. That migration needs to preserve existing launcher behavior while
allowing sibling repositories and future installed provider packs to join the
fleet without duplicating launcher logic.

The migration has two competing needs:

1. Preserve current local behavior so existing users and CI lanes keep working.
2. Create a reviewable path toward provider-first discovery, packaging, and
   eventually deletion of legacy configuration duplication.

Without an explicit rollout policy, new work can bypass the provider
abstractions, and compatibility shims can linger without clear exit criteria.

## Decision

We adopt three explicit discovery modes for launcher model discovery:

- `local-only`
  - Load only legacy `config/models.yaml` content.
  - Ignore external provider manifests.
  - Use this mode as the rollback-safe compatibility baseline.
- `hybrid`
  - Load legacy `config/models.yaml` first.
  - Merge external provider manifests afterward.
  - Preserve legacy duplicates when the same model id appears in both places.
  - This is the default migration mode.
- `provider-first`
  - Load provider manifests first.
  - Load legacy `config/models.yaml` only as a fallback for missing model ids.
  - Allow provider definitions to override duplicate legacy ids.
  - This mode is for parity testing and staged adoption, not yet the default.

The mode is controlled by `UPSTREAM_DRIFT_DISCOVERY_MODE`.

## Rollout Plan

### Phase 0: Documentation and explicit policy

- Publish this ADR and keep the epic order in issue #2395 as the canonical
  migration sequence.
- Keep `hybrid` as the default mode.

### Phase 1: Contract hardening

- Require provider packs to use versioned manifests.
- Standardize canonical identity, capability aliases, interchange artifacts,
  and provenance metadata.
- Keep legacy `models.yaml` as a compatibility adapter.

### Phase 2: Launcher/provider decoupling

- Move launcher discovery and tile assembly behind provider-aware services.
- Keep legacy config files and direct path assumptions only behind temporary
  adapters.
- Add focused CI regression tests around provider boundary modules so new work
  does not bypass the abstraction layer.

### Phase 3: Provider onboarding

- Onboard engine-model repos and shared utility providers in `hybrid` mode.
- Prove local and provider parity with compatibility and smoke-test coverage.
- Run `provider-first` in CI and internal validation before changing defaults.

### Phase 4: Deprecation and removal

- Change the default mode from `hybrid` to `provider-first` only after parity
  gates are green for the supported provider set.
- Remove duplicate legacy config paths and launcher-specific compatibility shims
  only after the exit criteria below are satisfied.

## Rollback Rules

- If provider discovery causes launcher regressions, set
  `UPSTREAM_DRIFT_DISCOVERY_MODE=local-only`.
- If provider parity is incomplete but external packs are still needed, revert
  to `hybrid`.
- Do not delete legacy configuration files or path adapters while any rollback
  depends on them.

## Exit Criteria for Deleting Legacy Shims

All of the following must be true before removing legacy local-path
assumptions or duplicate config sources:

1. Provider manifests exist for the supported engine/model repos in scope.
2. Compatibility tests cover local-only, hybrid, and provider-first discovery.
3. Launcher tile discovery no longer depends on bespoke local config parsing.
4. CI includes provider-boundary regression checks and provider smoke tests.
5. Installer and packaging profiles document how provider packs are discovered.
6. Provider-first mode has passed parity validation for the supported packs.

## Consequences

### Positive

- Migration state is explicit, reviewable, and reversible.
- Existing launcher behavior remains available while provider support grows.
- Provider-first adoption has clear prerequisites instead of ad hoc branch logic.
- Agents can implement migration work against a stable mode contract.

### Negative

- Temporary duplication remains during the compatibility window.
- The launcher will carry both legacy and provider-aware paths for a period.
- Some CI guardrails must stay scoped to migration-boundary modules until the
  broader launcher refactor lands.

## Validation

- `tests/unit/test_model_registry.py` covers `local-only`, `hybrid`, and
  `provider-first` discovery modes.
- `tests/launchers/test_provider_migration_boundaries.py` guards the provider
  boundary modules against reintroducing legacy repo-path shortcuts.
