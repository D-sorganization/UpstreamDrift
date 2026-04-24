# External Provider Onboarding

UpstreamDrift now discovers the first engine-model provider repos through a shared catalog:

- `MuJoCo_Models`
- `Drake_Models`
- `Pinocchio_Models`
- `OpenSim_Models`
- `Tools`
- `Movement-Optimizer`

## Local Clone Layout

Preferred development layout keeps the repos as siblings in one workspace:

```text
Repositories/
├── UpstreamDrift/
├── MuJoCo_Models/
├── Drake_Models/
├── Pinocchio_Models/
├── OpenSim_Models/
├── Tools/
└── Movement-Optimizer/
```

When a sibling repo contains `model_pack.yaml`, `model_pack.yml`, `.upstreamdrift/model_pack.yaml`, or `.upstreamdrift/model_pack.yml`, `UpstreamDrift` can discover it automatically in hybrid or provider-first discovery modes.

## Explicit Overrides

Set `UPSTREAM_DRIFT_PROVIDER_ROOTS` to add extra provider roots or override the default sibling layout.

Relative paths are resolved from the `UpstreamDrift` repo root. Multiple roots use the platform path separator.

## Runtime Behavior

- Missing provider repos do not break launcher startup.
- Provider-backed tiles appear when a provider manifest is present.
- If the provider repo exists but the engine runtime is not installed, launcher tiles downgrade to `runtime_unavailable`.
- If a provider manifest points at a missing source root, tiles degrade to `provider_unavailable`.
- Utility providers such as `Tools` and `Movement-Optimizer` are treated as launcher tools rather than engine packs, so they are not required to declare cross-engine identities or engine runtimes.

## Packaging Direction

This onboarding path is the bridge toward packaged provider distributions:

1. Keep provider-owned model metadata in the external repo manifests.
2. Let `UpstreamDrift` consume those manifests through the shared registry and launcher pipeline.
3. Preserve sibling-repo discovery for local development while installer profiles and packaged distributions are added in issue `#2401`.
