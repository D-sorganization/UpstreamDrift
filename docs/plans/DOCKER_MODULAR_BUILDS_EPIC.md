# EPIC: Modular Docker Builds & Professional Dependency Management

**Status:** Proposed
**Owner:** TBD
**Created:** 2026-05-19
**Tracking issue:** TBD (to be filed after this doc lands)

---

## 1. Problem statement

UpstreamDrift's current Docker pipeline is monolithic: a single `Dockerfile`
installs MuJoCo, Pinocchio, and the API stack into a ~3.5 GB image; a separate
`Dockerfile.heavy_test` best-effort-installs *every* engine and tops 8 GB; and
`docker-compose.yml` only knows about `runtime` and `training` targets.

Three concrete pain points follow from this:

1. **No user-selectable footprint.** A user who only wants the Pendulum +
   MuJoCo stack pays the same disk cost as a user who needs Drake and
   PyTorch+CUDA. The `validate_docker_stage()` helper in
   [`src/launchers/launcher_constants.py`](../../src/launchers/launcher_constants.py)
   advertises stages `all|mujoco|pinocchio|drake|base` that the Dockerfile
   does not actually define — the launcher UI offers builds that fail.
2. **Hard failures on missing dependencies.** Engine loaders in
   [`src/engines/loaders.py`](../../src/engines/loaders.py) raise
   `GolfModelingError` the moment an import fails. There is a probe layer
   ([`src/shared/python/engine_core/engine_probes.py`](../../src/shared/python/engine_core/engine_probes.py))
   that returns rich diagnostics, but it is not exposed to end users — the
   GUI sees a stack trace, not "Drake not installed; would you like to
   install it now?"
3. **No central capability source of truth.** Every layer (CLI, REST API,
   PyQt6 launcher, frontend) reimplements its own version of "is this
   feature available?" The result is drift: `pose_estimation/mediapipe_gui.py`
   uses MediaPipe with no probe; `pose_estimation/openpose_gui.py` calls
   the `OpenPoseProbe`; PyChrono is referenced in `pyproject.toml` as
   `[chrono]` extra but has no probe at all.

This epic delivers three coordinated capabilities to address these gaps.

## 2. Goals & non-goals

### Goals

- **G1 — Selectable Docker profiles.** Users pick from named presets
  (`slim`, `standard`, `research`, `biomech`, `full`, `gpu-training`) or
  compose arbitrary feature sets via build args
  (`FEATURES=mujoco,drake,pose-mediapipe`).
- **G2 — Graceful runtime degradation.** Every feature uses a probe; the
  launcher renders unavailable tiles in a disabled state with a tooltip
  that explains why and offers a one-click install.
- **G3 — Centralized capability registry.** A single Python module is the
  authoritative source for `(feature, probe, install hint, size estimate,
  docker stage)`; CLI, REST API, GUI, and CI all consume it.
- **G4 — Opt-in install workflow.** From a running application a user can
  trigger `pip install upstream-drift[drake]` (or the appropriate command
  per feature), watch progress, and have the registry hot-refresh.
- **G5 — Image size budgets enforced per profile.** Each named profile
  has a CI-enforced max size; regressions block merges.

### Non-goals

- **Replacing conda/mamba for OpenSim.** OpenSim and MyoSuite ship binary
  wheels but legacy installations still rely on conda; we treat conda as a
  supported install channel, not the default.
- **Cross-distro base image swaps.** We continue with `python:3.12-slim`
  (Debian). NVIDIA CUDA variants are produced by *layering on top* of the
  slim base, not by switching to `nvidia/cuda` as the base.
- **Native MATLAB packaging.** MATLAB licensing precludes redistribution;
  the MATLAB_3D engine remains host-only and is auto-disabled in all
  Docker profiles.
- **Reinventing the probe system.** We extend the existing probes; we do
  not rewrite them.

## 3. Current-state audit

### 3.1 Dockerfiles in the tree

| File | Purpose | Stages | Approx size |
|------|---------|--------|-------------|
| `Dockerfile` | Production API + Pinocchio | `builder`, `runtime`, `training` | runtime ~1.8 GB, training ~6 GB |
| `Dockerfile.heavy_test` | Local CI parity for `heavy_integration/` | single stage | ~8.5 GB |

The advertised launcher stages `all`, `mujoco`, `pinocchio`, `drake`,
`base` map to nothing in the actual Dockerfile.

### 3.2 Engine integration surface

| Engine | Loader | Probe | `pyproject` extra | Wheel? | Approx wheel size |
|--------|--------|-------|-------------------|--------|-------------------|
| MuJoCo | `load_mujoco_engine` | `MuJoCoProbe` | core | yes | ~120 MB |
| Drake | `load_drake_engine` | `DrakeProbe` | `[drake]` | yes (manylinux only) | ~700 MB |
| Pinocchio | `load_pinocchio_engine` | `PinocchioProbe` | `[pinocchio]` | yes (`pin`) | ~80 MB |
| OpenSim | `load_opensim_engine` | `OpenSimProbe` | `[biomechanics]` | conda preferred | ~400 MB |
| MyoSuite | `load_myosim_engine` | `MyoSimProbe` | `[biomechanics]` | yes | ~1.4 GB (depends on torch) |
| MATLAB_3D | `load_matlab_3d_engine` | `MatlabProbe` | host-only | no (licensed) | n/a |
| MediaPipe | (none — direct import) | **missing** | `[pose]` | yes | ~300 MB |
| OpenPose | (none — direct import) | `OpenPoseProbe` | external build | no | host-built |
| PyChrono | (none) | **missing** | `[chrono]` | conda only | ~600 MB |
| PyTorch CUDA | n/a | **missing** | `[rl]` + training stage | yes | ~2.8 GB |

### 3.3 Size contributions in the current `runtime` image (approximate)

```
python:3.12-slim base                ~120 MB
APT runtime libs (libgl, osmesa, …)  ~190 MB
venv: core pip stack                 ~340 MB
venv: mujoco                         ~120 MB
venv: pinocchio + pin + qpsolvers    ~210 MB
venv: pandas/matplotlib/sympy        ~180 MB
venv: sqlalchemy/bcrypt/cryptography ~90 MB
src/ + models (after copy)           ~600 MB
                                     -------
Total                                ~1.85 GB
```

The `training` stage adds:

```
+ torch cu124                        ~2.8 GB
+ gymnasium / sb3 / tb / ray         ~250 MB
                                     -------
Total                                ~5.0 GB
```

## 4. Target-state design

### 4.1 Capability registry (`src/shared/python/capabilities/`)

A new package whose **only** responsibility is to answer:

- *What features does this codebase support?*
- *Which features are usable in the current environment?*
- *If a feature is missing, what is the canonical install command?*
- *How big does each feature add to a Docker image?*

Public API:

```python
from src.shared.python.capabilities import (
    Capability,           # dataclass: name, probe, install_hint, size_mb, docker_stage, tier
    CapabilityStatus,     # enum: AVAILABLE | UNAVAILABLE | DEGRADED | UNKNOWN
    CapabilityReport,     # dataclass: capability, status, version, message, fix
    get_registry,         # returns the singleton CapabilityRegistry
    refresh,              # re-run all probes (after a pip install)
)

registry = get_registry()
report = registry.check("drake")
if report.status is not CapabilityStatus.AVAILABLE:
    print(report.fix.install_command)
```

The registry **wraps existing probes** in
`src/shared/python/engine_core/engine_probes.py`. It does not duplicate
probe logic. New probes are added there.

### 4.2 Modular `Dockerfile`

Replace the existing flat Dockerfile with a feature-flagged variant.
Selected features are passed as a single comma-separated build arg:

```bash
docker build --build-arg FEATURES=mujoco,drake,pose-mediapipe -t upstream-drift:custom .
```

Profile presets are defined in `docker/profiles.yaml`:

```yaml
profiles:
  slim:        { features: [api, pendulum] }                   # ~700 MB
  standard:    { features: [api, mujoco, pinocchio, pendulum] } # ~1.8 GB (current default)
  research:    { features: [standard, drake, pose-mediapipe] } # ~3.4 GB
  biomech:     { features: [standard, opensim, myosuite] }     # ~4.5 GB
  full:        { features: [research, biomech, chrono] }       # ~5.6 GB
  gpu-training:{ features: [full, torch-cuda, rl-stack] }      # ~9.0 GB
```

Implementation strategy uses one **conditional install stage per feature**
sharing a common builder. Each stage is a small RUN guard:

```dockerfile
ARG FEATURES=mujoco,pendulum
RUN python /opt/build/install_features.py "${FEATURES}"
```

`scripts/docker/install_features.py` reads the same profiles file the
registry uses, computes the union, and runs the pip installs. This keeps
the Dockerfile short and the *feature → pip args* mapping centralized in
one file.

GPU variants are layered on top with a second optional stage that
swaps the base image and re-installs torch with the `cu124` index.

### 4.3 Install-prompt UX

Three surfaces consume the registry:

1. **CLI** — `upstream-drift caps` prints a table; `upstream-drift caps
   install drake` runs the install in the current venv with progress on
   stdout.
2. **REST API** — `GET /api/capabilities` returns the full report;
   `POST /api/capabilities/install/{name}` triggers an install (gated by
   `GOLF_AUTH` and only available when not running as a non-root container
   user — Docker-image users are directed to rebuild instead).
3. **PyQt6 launcher** — `MissingDependencyDialog` opens when a tile is
   clicked whose primary capability is unavailable. It shows the diagnostic
   message, the install command, an "Install now" button, and a "Build a
   bigger Docker image" link to `docs/development/DOCKER_SETUP.md`.

### 4.4 CI gates

- `docker-size-gates.yml` is extended to build and size every named
  profile. Each profile has a per-MB budget in `docker/profiles.yaml`
  (`max_size_mb`).
- `docker-smoke.yml` is added to run `python -m
  src.shared.python.capabilities.registry --check` inside every profile
  image and assert that the expected capabilities are AVAILABLE *and*
  that capabilities outside the profile are UNAVAILABLE (preventing
  accidental dep leakage between profiles).

## 5. Phasing

### Phase 1 — Foundation (this PR, `feat/modular-docker-builds`)

- Capability registry package + tests.
- New probes: `MediaPipeProbe`, `PyChronoProbe`.
- `/api/capabilities` endpoint (read-only).
- `upstream-drift caps` CLI command (read-only).
- Documentation: this epic + reference doc at
  `docs/operations/capability-registry.md`.

**Acceptance:** registry reports correct status for all 11 capabilities in
both bare-metal dev environments and the current `Dockerfile` runtime
image; no behavioral change to existing loaders.

### Phase 2 — Modular Dockerfile + profile presets

- `docker/profiles.yaml` source of truth.
- `scripts/docker/install_features.py` consumed by Dockerfile.
- New `Dockerfile` (replacing current) with feature ARGs.
- `docker-compose.profiles.yml` overlay (one service per profile).
- Launcher `validate_docker_stage()` reads from `profiles.yaml`.
- CI: profile size matrix in `docker-size-gates.yml`.

**Acceptance:** `slim` profile <800 MB, `standard` ≈ current size,
`research` <3.5 GB, `full` <6 GB; all profiles boot and `/health`
responds; profile capability assertions in CI pass.

### Phase 3 — Install-prompt UX

- `MissingDependencyDialog` PyQt6 widget.
- `POST /api/capabilities/install/{name}` (auth-gated, env-gated).
- Launcher tiles route through the dialog when their capability is not
  AVAILABLE.
- Hot-refresh of the registry after a successful install (no process
  restart needed).
- Documentation: `docs/user_guide/installing_optional_features.md`.

**Acceptance:** clicking a Drake-dependent tile on a Drake-less venv pops
the dialog; clicking "Install" results in Drake being importable and the
tile becoming clickable, all without restart; install is rejected with a
clear message inside containers (rebuild instead).

### Phase 4 — Hardening & polish (post-MVP, separate issues)

- Add `conda env create` parity for OpenSim/PyChrono.
- Generate per-profile SBOM via `docker sbom` in CI.
- Per-profile vulnerability scanning gates.
- Self-service "image builder" web page in the launcher's settings tab.

## 6. Acceptance criteria (epic-level)

- [ ] A user can run `docker build --build-arg PROFILE=slim` and get an
      image under 800 MB.
- [ ] A user can run `docker build --build-arg FEATURES=mujoco,drake` and
      get a working image with exactly those engines.
- [ ] On a host where `drake` is not installed, opening any Drake-backed
      tile in the launcher shows a dialog with the install command, never
      a stack trace.
- [ ] `GET /api/capabilities` returns a JSON document covering every
      engine, pose backend, and ML stack listed in §3.2.
- [ ] `docker-size-gates.yml` fails if any profile exceeds its budget.
- [ ] `docker-smoke.yml` fails if a profile contains a dep outside its
      declared feature set.
- [ ] `MEMORY.md` is updated with the canonical install commands and
      profile budgets so future agents can extend the system without
      re-discovering them.

## 7. Risks & mitigations

| Risk | Mitigation |
|------|------------|
| OpenSim/PyChrono wheels are platform-fragile and may regress | Treat them as `experimental` tier; CI green requires `core`+`extended` only |
| Drake wheel size means even `research` profile is 3+ GB | Document this clearly; surface size at build time so users aren't surprised |
| Install-from-running-app could break a user's venv | Run installs with `--user` when not in a venv; refuse inside the image's non-root user; always print the exact command first |
| Layer cache invalidation when profiles change | Profile-specific stages share a base; install_features.py uses sorted feature lists so identical sets hit the same cache |
| Capability hot-refresh stale due to Python import caching | `refresh()` does `importlib.invalidate_caches()` and reloads the probe module; document that some packages (e.g. `mujoco` with native libs) still need a restart |

## 8. Child issues (to be filed when this epic is approved)

1. **Phase 1 — Capability registry & probes** — create `src/shared/python/capabilities/`, add MediaPipe and PyChrono probes, `/api/capabilities` endpoint, CLI `caps` command.
2. **Phase 2.a — `docker/profiles.yaml` + `install_features.py`** — single source of truth for feature → install commands and Docker stage mapping.
3. **Phase 2.b — Modular Dockerfile rewrite** — feature ARGs, profile presets, deprecate `Dockerfile.heavy_test` in favor of `Dockerfile --build-arg PROFILE=full`.
4. **Phase 2.c — `docker-compose.profiles.yml`** — compose overlay; launcher reads profile list dynamically.
5. **Phase 2.d — CI size matrix** — extend `docker-size-gates.yml` to build all profiles with per-profile budgets.
6. **Phase 2.e — CI capability smoke test** — `docker-smoke.yml` asserting profile membership both positively and negatively.
7. **Phase 3.a — `MissingDependencyDialog`** — PyQt6 widget + tests with `pytest-qt`.
8. **Phase 3.b — Install API** — `POST /api/capabilities/install/{name}` with auth & env guards.
9. **Phase 3.c — Launcher integration** — every tile routes through the dialog when its capability is unavailable.
10. **Phase 3.d — Hot-refresh** — registry reload after install; tests covering the round-trip.
11. **Phase 4 — Conda parity, SBOM, vuln scanning, image-builder UI** — bundled into a follow-on epic.

## 9. References

- Current Dockerfile: [`Dockerfile`](../../Dockerfile)
- Heavy test Dockerfile: [`Dockerfile.heavy_test`](../../Dockerfile.heavy_test)
- Compose: [`docker-compose.yml`](../../docker-compose.yml)
- Engine tiers: [`docs/engines/support_tiers.md`](../engines/support_tiers.md)
- Probe system: [`src/shared/python/engine_core/engine_probes.py`](../../src/shared/python/engine_core/engine_probes.py)
- Loaders: [`src/engines/loaders.py`](../../src/engines/loaders.py)
- Docker-first plan (historical): [`docs/plans/DOCKER_FIRST_ADOPTION_PLAN.md`](DOCKER_FIRST_ADOPTION_PLAN.md)
- Existing CI size gate: [`.github/workflows/docker-size-gates.yml`](../../.github/workflows/docker-size-gates.yml)
