# Capability registry — operations reference

The capability registry is UpstreamDrift's single source of truth for
"is this feature available *right now* in this environment, and if
not, how do I install it?" It is consumed by the CLI, the REST API,
the PyQt6 launcher, and CI smoke tests.

## When to use which entry point

| Surface | Command / endpoint | Use case |
|---------|--------------------|----------|
| CLI table | `python -m src.shared.python.feature_registry` | Local development; "what do I have?" |
| CLI JSON | `python -m src.shared.python.feature_registry --json` | Scripting; CI assertions |
| CLI single feature | `python -m src.shared.python.feature_registry --check drake` | Probe one feature; exit status reflects availability |
| REST list | `GET /api/v1/capabilities` | UIs, dashboards |
| REST single | `GET /api/v1/capabilities/{name}` | Tooltips, hover-cards |
| REST refresh | `POST /api/v1/capabilities/refresh` | After an external `pip install` |
| REST install | `POST /api/v1/capabilities/{name}/install` | Opt-in install from a GUI |

## Adding a new feature

Three coordinated edits:

1. **Append a `Feature(...)` entry to `src/shared/python/feature_registry/features.py`.** Pick a stable lowercase `name`, fill the `install_command`, the `docker_stage` it belongs to, and a rough `approx_size_mb`.
2. **Register a probe in `src/shared/python/feature_registry/probes.py`.** Either reuse an existing engine probe from `src/shared/python/engine_core/engine_probes.py` or add a small `_probe_<name>` function returning a `ProbeOutcome`.
3. **(Optional) Mention the feature in `docker/profiles.yaml`.** If the feature is part of a named build profile, add it there.

The registry's import-time invariant check ensures that every feature with a `probe_key` has a registered probe — the package won't import if the mapping is out of sync.

## Docker build profiles

Profiles compose features via `extends:`:

```yaml
research:
  description: Standard plus Drake and MediaPipe.
  extends: standard
  features: [drake, pose-mediapipe]
  max_size_mb: 3500
```

Build a profile:

```bash
docker build --build-arg PROFILE=research \
             -f Dockerfile.modular \
             -t upstream-drift:research .
```

Build a custom feature set:

```bash
docker build --build-arg FEATURES=mujoco,drake,pose-mediapipe \
             -f Dockerfile.modular \
             -t upstream-drift:custom .
```

The translation from feature → pip command lives in `scripts/docker/install_features.py`, which reads `profiles.yaml` *and* the registry's `features.py`. There is no other place the install command may live.

## Opt-in install from a running app

`install_feature("drake")` shells out to `pip install upstream-drift[drake]` using the active interpreter (`sys.executable -m pip`), so the install lands in the venv that's running the call.

Safety rails (see `installer.py`):

- Refuses to run inside a non-root Docker container — points the user at the Docker rebuild path instead. Rebuilding is the right answer because an image is the contract; mutating its venv silently diverges from the declared profile.
- Refuses `external`-channel features (OpenPose) — surfaces the documented build-from-source instructions.
- Refuses `conda`-channel features when `conda` is not on `PATH`.

After a successful install, the registry is refreshed (probes re-run, `importlib.invalidate_caches()` called) so the new state is visible without a process restart.

## Tier alignment

| Registry tier | Engine tier | Examples | CI guarantee |
|---------------|-------------|----------|--------------|
| `core` | `core` | MuJoCo, API, pendulum | Required PR CI |
| `extended` | `extended` | Drake, Pinocchio | Nightly cross-engine |
| `experimental` | `experimental` | OpenSim, MyoSuite, PyChrono | Best-effort |
| `tooling` | n/a | MediaPipe, OpenPose, PyTorch CUDA, RL stack | None — opt-in |

See [`docs/engines/support_tiers.md`](../engines/support_tiers.md) for the tier policy.

## Related files

- Epic: [`docs/plans/DOCKER_MODULAR_BUILDS_EPIC.md`](../plans/DOCKER_MODULAR_BUILDS_EPIC.md)
- Probes: [`src/shared/python/engine_core/engine_probes.py`](../../src/shared/python/engine_core/engine_probes.py)
- Features: [`src/shared/python/feature_registry/features.py`](../../src/shared/python/feature_registry/features.py)
- Registry: [`src/shared/python/feature_registry/registry.py`](../../src/shared/python/feature_registry/registry.py)
- Installer: [`src/shared/python/feature_registry/installer.py`](../../src/shared/python/feature_registry/installer.py)
- API route: [`src/api/routes/capabilities.py`](../../src/api/routes/capabilities.py)
- Profiles: [`docker/profiles.yaml`](../../docker/profiles.yaml)
- Compose overlay: [`docker-compose.profiles.yml`](../../docker-compose.profiles.yml)
- Dockerfile: [`Dockerfile.modular`](../../Dockerfile.modular)
- Translator: [`scripts/docker/install_features.py`](../../scripts/docker/install_features.py)
