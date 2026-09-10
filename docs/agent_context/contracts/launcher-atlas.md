# Launcher Registries to Capability Atlas

## Responsibilities

The existing model, launcher metadata and feature-parity registries own product
inventory. `scripts.capability_atlas.model.build` projects them into the product
atlas. The agent catalog links these authorities instead of copying tile lists.

## Data Contract

Read launcher model entries from `src/config/models.yaml`, web metadata from
`launcher_manifest.json`, and feature contracts from `feature_parity.json`.
Preserve identifiers and the `parity`, `gap` and `exempt` distinctions.
Curated architecture connections live in `capability_connections.json`.
Registry metadata is not a runtime health or scientific approval signal.

The same connections authority now contains `capture_goals` with schema
`capture-goals/1`. The atlas uses `CaptureGoalCatalog` and `validate_bindings`
to check unique identifiers, known nodes and workflow keys, valid navigation
actions and acyclic prerequisites. Its saved plans carry a catalog revision;
the capture wizard evaluates current inputs when opening them. Architecture
data-flow edges remain distinct from executable goal prerequisites.

The registry now includes model-only analysis, comparison drawings, metric
reference readouts, Trace v2 imports and Pose Studio world references. Preserve
their explicit desktop-only status and evidence paths; these entries do not
establish web parity or scientific qualification. The existing
[Shared Analysis Contracts](../../architecture/SHARED_ANALYSIS_CONTRACTS.md)
describe the underlying sampling and geometry boundaries.

## Lifecycle and Failures

Run `python3 -m scripts.generate_capability_atlas` after relevant registry
changes and its `--check` mode in validation. Refer to the actual registration
when an external or virtual tile has no local source file. Broken schema or
missing workflow declarations must fail generation rather than produce a partial
map that appears complete.

## Evidence

- [Atlas Builder](../../../scripts/capability_atlas/model.py)
- [Atlas Generation Tests](../../../tests/scripts/test_capability_atlas.py)
- [Existing Product Map](../../architecture/CAPABILITY_ATLAS.md)

## Rationale

Product inventory belongs to the existing application registries. A second
manual catalog of every tile would drift. Development context adds public
interfaces and integration evidence around that existing inventory.
