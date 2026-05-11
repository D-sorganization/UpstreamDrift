# ADR 0014: Shared Biomechanics Models

## Status
Accepted

## Context
The `UpstreamDrift` repository serves as the central hub for biomechanical modeling and simulation, but the actual domain-specific models (e.g. OpenSim, MuJoCo, Drake, Pinocchio) reside in their own sibling repositories (e.g. `OpenSim_Models`, `MuJoCo_Models`). Historically, we duplicated XML and URDF files directly into the `UpstreamDrift` tree, which caused inevitable drift between the source of truth and our test infrastructure.

## Decision
We establish a unified, four-tier resolution convention for all sibling biomechanics repositories:

1. **Editable Sibling Checkout**: If a sibling repo exists adjacent to `UpstreamDrift` on disk, use its live `model_pack:resolve()` function.
2. **Pip-Installed Sibling**: If a sibling is installed into the active Python environment (via wheel or editable pip), use its `model_pack:resolve()` hook.
3. **Vendored Snapshot**: CI pipelines and default headless environments resolve against hermetic, tagged snapshots located in `vendor/biomech-models/<repo>`.
4. **Environment Variable Override**: Users can explicitly bind a repo path using `*_MODELS_HOME` or `*_OPTIMIZER_HOME`.

The resolution logic is encapsulated in `src.shared.python.config.model_source_providers` via the `@register_source` decorator.

## Consequences
- **Positive**: Single source of truth. Models are updated once in their canonical repository. CI runs are hermetic via the vendor snapshot.
- **Positive**: Simplifies local development; developers just need to clone the sibling repositories next to `UpstreamDrift`.
- **Negative**: Requires maintaining a new `scripts/update_biomech_vendor.py` snapshot tool.
