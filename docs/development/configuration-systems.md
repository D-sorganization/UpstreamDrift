# Configuration Systems

UpstreamDrift has one owner for each class of configuration. Do not add new
root-level `config/` or `configs/` directories.

## Runtime Settings

Runtime settings belong in `src/shared/python/config/` and should be exposed
through typed accessors or settings models. Environment variables take
precedence over YAML defaults and hard-coded fallbacks.

Use this home for API host/port, auth toggles, deployment environment, secrets,
and application settings that are read while the process starts.

## Launcher Manifests

Launcher tile manifests remain in `src/config/` because both the Python launcher
and web/Tauri surfaces read them as product manifests rather than runtime
settings.

Use this home for `models.yaml`, `launcher_manifest.json`, and related loader
contracts.

## Domain-Owned YAML

Domain seed files live under the package that owns the domain behavior.

- BunkerShot3D calibration presets: `src/bunkershot3d/calibration/configs/`
- UX field and error metadata: `src/shared/python/ux/config/`

Code should resolve these files through package-relative constants or helper
functions, not by rebuilding repository-root paths at call sites.

## CI And Governance Policy

CI baselines, budgets, waivers, and policy files live in `scripts/config/`.
These files are not runtime application settings; they are repository governance
inputs consumed by scripts and workflows.

Examples include file-size budgets, module-size baselines, architecture-debt
policy, pip-audit waivers, mypy baselines, and documentation-size budgets.

## Adding New Configuration

1. Choose the owner from the categories above before creating a file.
2. Add a typed loader or path constant next to the owning code.
3. Add a focused test that loads the checked-in config and validates the schema.
4. Update this document if the new config class does not fit an existing
   category.
5. Keep generated mirrors, such as TypeScript metadata registries, regenerated in
   the same PR.

Configuration files should have exactly one source of truth. Compatibility
shims require a tracked issue and an expiry date.
