# Biomech Workspace Setup

> Companion to [`docs/adr/0014-shared-biomech-models.md`](adr/0014-shared-biomech-models.md).
> Closes the user-facing portion of
> [UpstreamDrift#5184](https://github.com/D-sorganization/UpstreamDrift/issues/5184).

UpstreamDrift discovers models from the five sibling biomechanics repos:

| Sibling repo         | Python package       | Env-var override          |
| -------------------- | -------------------- | ------------------------- |
| `MuJoCo_Models`      | `mujoco_models`      | `MUJOCO_MODELS_HOME`      |
| `Drake_Models`       | `drake_models`       | `DRAKE_MODELS_HOME`       |
| `Pinocchio_Models`   | `pinocchio_models`   | `PINOCCHIO_MODELS_HOME`   |
| `OpenSim_Models`     | `opensim_models`     | `OPENSIM_MODELS_HOME`     |
| `Movement-Optimizer` | `movement_optimizer` | `MOVEMENT_OPTIMIZER_HOME` |

For each sibling, the discovery layer walks four tiers in order. The
first tier that resolves wins.

1. **Editable checkout** at `../<RepoName>/` (next to your UpstreamDrift
   checkout). Detected by the presence of `pyproject.toml`.
2. **Pip-installed package** — calls `<pkg>.model_pack:resolve()` (or
   `<pkg>.tool_pack:resolve()` for `Movement-Optimizer`).
3. **Vendored snapshot** at `vendor/biomech-models/<RepoName>/`,
   committed by `scripts/update_biomech_vendor.py`.
4. **Env-var override** — set `<REPO>_HOME` to an absolute path.

## Editable mode (recommended for development)

Clone the siblings next to your UpstreamDrift checkout:

```
workspace/
  UpstreamDrift/
  MuJoCo_Models/
  Drake_Models/
  Pinocchio_Models/
  OpenSim_Models/
  Movement-Optimizer/
```

Then run the bootstrap script. It performs `pip install -e ../<RepoName>`
for every sibling that exists and silently skips the rest:

```bash
./scripts/setup_biomech_workspace.sh
```

The script prints a summary listing which repos it installed, skipped,
or failed on. A failed install propagates a non-zero exit code so CI
won't quietly continue.

Verify the wiring:

```bash
python3 -c "from src.shared.python.config.model_source_providers import resolve_all_siblings; \
           [print(name, r.tier, r.models_root) for name, r in resolve_all_siblings().items()]"
```

## Vendored mode (default for CI)

When no sibling checkout exists, the discovery layer falls through to
`vendor/biomech-models/<RepoName>/`. Snapshots are produced from
tagged releases:

```bash
python3 scripts/update_biomech_vendor.py --repo MuJoCo_Models --ref v1.4.0
python3 scripts/update_biomech_vendor.py --repo Drake_Models --ref v0.3.2
```

Each snapshot includes the manifest file (`model_pack.yaml` or
`tool_pack.yaml`) and a `models/` tree. A `VENDOR_PROVENANCE.txt`
marker records the source URL and ref.

Snapshots are committed to UpstreamDrift; CI uses them and never clones
the sibling repos at test time.

## Env-var override

For unusual layouts (network-mounted models, custom forks), set the
override:

```bash
export MUJOCO_MODELS_HOME=/srv/biomech/mujoco_models
```

Tier 4 wins only when none of the first three tiers find a usable
source.

## Pytest fixture

The pytest suite exposes a `--biomech-mode` option mirroring the
existing `--tools-mode` flag:

```bash
python3 -m pytest --biomech-mode=editable   # explicit editable
python3 -m pytest --biomech-mode=vendored   # explicit vendored
python3 -m pytest --biomech-mode=env        # explicit env-var
python3 -m pytest                           # auto-detect (editable if any
                                            #  sibling checkout exists,
                                            #  else vendored)
```

The active mode is exposed as the `biomech_mode` fixture.

## Diagnostics

`launcher_diagnostics.py` includes a biomech-sibling check that prints
the resolution tier and (if reachable) the `model_pack.yaml` schema
version for each sibling.

```bash
python3 -m src.launchers.launcher_diagnostics
```

## See also

- [Motion Pipeline guide](motion_pipeline/README.md) — downstream
  consumer of the resolved models.
- [ADR-0014](adr/0014-shared-biomech-models.md) — the convention and
  four-tier resolution order.
- [`CLAUDE.md` → Cross-Repo Dependencies](../CLAUDE.md) — the
  established `Tools` precedent this builds on.
