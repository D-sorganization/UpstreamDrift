# ADR-0014: Shared Biomech Model-Pack Convention

- Status: Accepted
- Date: 2026-05-11
- Decision Makers: UpstreamDrift core maintainers
- Related Issues/PRs:
  Umbrella
  [#5179](https://github.com/D-sorganization/UpstreamDrift/issues/5179),
  this ADR closes the design portion of
  [#5184](https://github.com/D-sorganization/UpstreamDrift/issues/5184).
  Coordination on the publishing side:
  [MuJoCo_Models#266](https://github.com/D-sorganization/MuJoCo_Models/issues/266),
  [Drake_Models#240](https://github.com/D-sorganization/Drake_Models/issues/240),
  [Pinocchio_Models#282](https://github.com/D-sorganization/Pinocchio_Models/issues/282),
  [OpenSim_Models#264](https://github.com/D-sorganization/OpenSim_Models/issues/264),
  [Movement-Optimizer#456](https://github.com/D-sorganization/Movement-Optimizer/issues/456).

## Context

UpstreamDrift embeds biomechanics model assets in-tree under
`src/shared/models/myosuite/` and `src/shared/models/opensim/opensim-models/`.
The five sibling biomechanics repos (`MuJoCo_Models`, `Drake_Models`,
`Pinocchio_Models`, `OpenSim_Models`, `Movement-Optimizer`) each ship the
same kind of content from their own repos. Today there is no shared
contract:

- OpenSim XML drifts between `OpenSim_Models/` and the copy in
  `UpstreamDrift/src/shared/models/opensim/opensim-models/`. Drift is
  inevitable because the two trees evolve on different PRs.
- Adding an exercise in `MuJoCo_Models` requires a follow-up sync PR
  in UpstreamDrift.
- UpstreamDrift CI cannot validate that a given exercise's MuJoCo and
  Drake variants agree, because UpstreamDrift has no way to find
  `Drake_Models` on disk.

We already established a precedent for cross-repo dependencies with the
`Tools` integration (`vendor/ud-tools/` + `scripts/setup_tools_workspace.sh` +
`--tools-mode` fixture). The biomechanics fleet has the same shape — five
peer repos, three execution contexts (local dev, CI, end users) — so we
extend that precedent rather than invent a new pattern.

## Decision

We adopt a **shared model-pack convention** with three publishing-side
artefacts and a four-tier consumer-side resolution order.

### Publishing side (sibling repos)

Each sibling biomech repo publishes:

1. A canonical models tree at `<repo>/models/` or
   `<repo>/src/<pkg>/models/` (picked per repo in its coordination issue).
2. A `model_pack.yaml` manifest (or `tool_pack.yaml` for
   `Movement-Optimizer`) at the repo root that conforms to the
   `model_pack/v1` JSON Schema published by UpstreamDrift at
   `src/shared/python/biomech/schemas/model_pack_v1.json`.
3. A Python entry point — `<pkg>.model_pack:resolve()` —
   that returns the absolute path of the models tree to in-process
   callers.

The schema covers two variants:

- **Model packs** declare `engine`, `engine_version`, `anthropometrics`,
  `format`, `models_root`, and an `exercises[]` array of `{id, path}`
  pairs. Optional fields: `axis_convention`, `addons`.
- **Tool packs** describe optimisers/analysers that consume models from
  other packs. They declare `role`, `formulation`, `muscle_model`,
  `plane`, `links`, `supported_exercises[]`,
  `consumes_models_from[]`, and `produces[]`.

### Consumer side (UpstreamDrift)

UpstreamDrift resolves each sibling via the following precedence,
implemented by `_resolve_sibling()` in
`src/shared/python/config/model_source_providers.py`:

1. **Editable sibling checkout** — `../<RepoName>/` next to the
   UpstreamDrift checkout. Detected by the presence of `pyproject.toml`
   (we deliberately do not `import` the sibling, since the entry point
   may not yet be wired up). The manifest's declared `models_root` is
   honoured; if absent, conventional locations are tried.
2. **Pip-installed sibling package** — `<pkg>.model_pack:resolve()` (or
   `tool_pack:resolve()` for tool packs) is called.
3. **Vendored snapshot** at `vendor/biomech-models/<RepoName>/`,
   committed via `scripts/update_biomech_vendor.py --repo ... --ref ...`.
4. **Env-var override** — `<REPO>_HOME` (e.g. `MUJOCO_MODELS_HOME`) —
   for power users and ad-hoc layouts.

The five sibling-specific providers (`mujoco_models_source`,
`drake_models_source`, `pinocchio_models_source`, `opensim_models_source`,
`movement_optimizer_source`) all delegate to the shared helper. Each is
decorated with `@register_source(...)` so the launcher diagnostics can
enumerate them.

Bootstrapping the editable mode is done with
`scripts/setup_biomech_workspace.sh`, which pip-installs every sibling
that exists at `../<RepoName>/` and skips the rest silently. CI keeps
working without any sibling checkout because the vendored snapshot tier
covers it.

## Alternatives Considered

1. **Git submodules.** Mirrors the current `vendor/ud-tools` model but
   has the same drift problem — sibling repos move on tagged releases,
   not arbitrary commits, and submodule UX is poor for non-power-users.
   Vendored snapshots committed by a script are simpler.
2. **Single monorepo.** Would eliminate the discovery problem but
   breaks the five sibling repos' independent release cycles and would
   require coordinated migration of their CI. Out of scope for #5184.
3. **Pip-install only.** Skipping the editable-checkout tier would
   force every contributor to publish + re-install on every change.
   The editable tier is essential for iterative development.
4. **Per-engine resolvers without a shared schema.** Tried in earlier
   drafts; produced five copies of the same resolution logic plus five
   ad-hoc YAML shapes. Replaced by `_resolve_sibling` + the v1 schema.

## Consequences

- Positive:

  - One published schema (`model_pack_v1.json`) governs every sibling.
  - Editable / installed / vendored / env tiers cover every dev and CI
    layout we currently see.
  - Removing the in-tree copies under `src/shared/models/` becomes a
    follow-up PR, not a coordinated drop.
  - Adding a new sibling repo means: add a `_SiblingSpec`, register a
    decorated provider, and re-run the bootstrap script.

- Negative:

  - Until every sibling publishes its `model_pack.yaml`, the resolver
    falls back to conventional paths. This is by design (graceful
    rollout) but means CI cannot enforce manifests yet.
  - Vendored snapshots add disk footprint to UpstreamDrift checkouts.
    The snapshots live under `vendor/biomech-models/` and can be
    cleaned with `git rm` once a sibling publishes a stable wheel.

- Follow-ups:
  - Delete `src/shared/models/opensim/opensim-models/` once the
    `opensim_models` provider is exercised in green CI.
  - Decide whether `MyoSuite` (currently at
    `src/shared/models/myosuite/`) warrants its own provider.
  - Hook a CI step that validates every sibling's published manifest
    against `model_pack_v1.json` after the publishing-side issues land.

## Validation

- `src/shared/python/biomech/schemas/model_pack_v1.json` is exercised by
  `tests/test_model_pack_schema.py`, covering valid model packs for each
  engine, a valid tool pack, and a representative set of invalid
  payloads.
- `tests/test_model_source_providers.py` verifies the four-tier
  resolution order for each of the five sibling providers with mocked
  filesystem and importlib state.
- `tests/test_setup_biomech_workspace.py` smoke-tests the bootstrap
  shell script against synthesised sibling checkouts.
- `src/launchers/launcher_diagnostics.py` reports the active resolution
  tier and the `model_pack.yaml` schema version for each sibling at
  launcher startup.
