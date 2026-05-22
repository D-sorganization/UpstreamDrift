# Installing optional features

UpstreamDrift ships with a small core (API + pendulum + MuJoCo) and a
larger set of optional engines and tools you install only when you
need them. This guide shows the four ways to add a feature and when
to pick each one.

## Quick answer

| You want to …                              | Use this                                       |
| ------------------------------------------ | ---------------------------------------------- |
| Add a feature to your local Python install | `pip install 'upstream-drift[<extra>]'`        |
| Add it from inside a running launcher      | Click the tile — a dialog offers to install    |
| Add it to a Docker image                   | Rebuild with a profile that includes it        |
| See what's installed right now             | `python -m src.shared.python.feature_registry` |

## The feature catalog at a glance

| Feature                      | Extra                       | Channel         | Approx size          |
| ---------------------------- | --------------------------- | --------------- | -------------------- |
| MuJoCo (`mujoco`)            | (in core)                   | pip             | 120 MB               |
| Drake (`drake`)              | `[drake]`                   | pip             | 700 MB               |
| Pinocchio (`pinocchio`)      | `[pinocchio]`               | pip             | 210 MB               |
| OpenSim (`opensim`)          | `[biomechanics]` (or conda) | conda preferred | 400 MB               |
| MyoSuite (`myosuite`)        | `[biomechanics]`            | pip             | 1.4 GB (incl. torch) |
| PyChrono (`chrono`)          | `[chrono]` (conda only)     | conda           | 600 MB               |
| MediaPipe (`pose-mediapipe`) | `[pose]`                    | pip             | 300 MB               |
| OpenPose (`pose-openpose`)   | n/a                         | external build  | —                    |
| PyTorch CUDA (`torch-cuda`)  | n/a                         | pip             | 2.8 GB               |
| RL stack (`rl-stack`)        | `[rl]`                      | pip             | 250 MB               |

Sizes are rough wheel + native lib estimates used for Docker profile
budgets — your local pip install may add transitive deps that bring
the total higher.

## Method 1 — `pip install` directly

For local development, install the feature in your active venv:

```bash
pip install 'upstream-drift[drake]'           # one extra
pip install 'upstream-drift[drake,pose]'      # multiple at once
pip install 'upstream-drift[all-engines]'     # all rigid-body engines
```

Conda-channel features (OpenSim, PyChrono) install best from
conda-forge:

```bash
conda install -c opensim-org opensim
conda install -c projectchrono pychrono
```

External builds (OpenPose) follow the upstream project's build
instructions — UpstreamDrift just probes for the resulting Python
module.

After installing, restart the launcher (or call
`POST /api/v1/capabilities/refresh`) so the registry sees the new
state.

## Method 2 — In-launcher install dialog

When you click a launcher tile whose backing engine isn't installed,
the **InstallPromptDialog** (in
[`src/shared/python/ui/dialogs/install_prompt.py`](../../src/shared/python/ui/dialogs/install_prompt.py))
opens. It offers three choices:

- **Yes, install** — runs the documented install command in a
  background thread; output streams into the dialog; the registry is
  refreshed automatically on completion.
- **Not now** — closes the dialog without installing. You'll be
  prompted again next time you open the tile.
- **Don't ask again** — closes and persists the suppression to
  `~/.upstreamdrift/prefs.json` so this feature won't prompt again on
  this machine.

### When the dialog refuses or skips

A few cases the dialog will _not_ run the install for you:

| Situation                                       | What happens                                          | Why                                                            |
| ----------------------------------------------- | ----------------------------------------------------- | -------------------------------------------------------------- |
| Running inside an unprivileged Docker container | Refuses with a "rebuild with a larger profile" hint   | The image is the contract — see Method 3 below.                |
| `conda`-only feature with no `conda` on PATH    | Refuses with the documented manual command            | Manual `conda install` is the supported path.                  |
| OpenPose                                        | Refuses — "external-build only"                       | Requires CUDA + cmake; follow upstream docs.                   |
| Feature already in _Don't ask again_            | Dialog does not open; `prompt()` returns `SUPPRESSED` | Clear the entry in `~/.upstreamdrift/prefs.json` to re-enable. |

The dialog always shows the exact command first so you can copy and
run it yourself if you prefer.

## Method 3 — Rebuild a Docker image with a different profile

Inside Docker, the right way to add features is to rebuild from a
larger profile rather than mutate the running container's venv. The
profile catalog ([`docker/profiles.yaml`](../../docker/profiles.yaml)):

| Profile        | Includes                  | Budget |
| -------------- | ------------------------- | ------ |
| `slim`         | API + pendulum only       | 900 MB |
| `standard`     | + MuJoCo + Pinocchio      | 2.2 GB |
| `research`     | + Drake + MediaPipe       | 3.5 GB |
| `biomech`      | + OpenSim + MyoSuite      | 4.8 GB |
| `full`         | + PyChrono                | 6.0 GB |
| `gpu-training` | + PyTorch CUDA + RL stack | 9.5 GB |

Build a named profile:

```bash
docker build --build-arg PROFILE=research \
             -f Dockerfile.modular \
             -t upstream-drift:research .
```

Or compose an arbitrary feature set:

```bash
docker build --build-arg FEATURES=mujoco,drake,pose-mediapipe \
             -f Dockerfile.modular \
             -t upstream-drift:custom .
```

The compose overlay `docker-compose.profiles.yml` defines one service
per profile on a distinct port so you can run multiple side-by-side:

```bash
docker compose -f docker-compose.yml -f docker-compose.profiles.yml \
    --profile research up backend-research
```

Profile size budgets are enforced by
[`.github/workflows/docker-size-gates.yml`](../../.github/workflows/docker-size-gates.yml);
profile-membership assertions live in
[`.github/workflows/docker-smoke.yml`](../../.github/workflows/docker-smoke.yml).

## Method 4 — Programmatic checks

For scripts and CI:

```bash
# Whole table:
python -m src.shared.python.feature_registry

# JSON for scripting:
python -m src.shared.python.feature_registry --json

# Single feature; exit status reflects availability:
python -m src.shared.python.feature_registry --check drake
echo $?   # 0 == available, 1 == missing
```

REST API for dashboards and remote tooling:

```bash
curl http://localhost:8001/api/v1/capabilities
curl http://localhost:8001/api/v1/capabilities/drake
curl -X POST http://localhost:8001/api/v1/capabilities/refresh
```

Python convenience re-export (canonical import path for the registry):

```python
# Both work — the second is a thin re-export shim.
from src.shared.python.feature_registry import get_registry
from src.core.capability_registry import get_registry
```

## Troubleshooting

### "I just installed it but the launcher still says missing"

Run `POST /api/v1/capabilities/refresh` (or restart the launcher). The
registry caches its first probe to keep tooltips fast; the refresh
endpoint re-runs every probe and invalidates Python's import caches.

### "The dialog won't let me install in Docker"

That's intentional. Mutating a running container's venv silently
diverges the image from its declared profile and breaks reproducibility.
Rebuild with a larger profile (Method 3) — `docker compose down &&
docker compose up --build` is usually enough.

### "I want this feature in `slim` but the dialog won't let me run conda"

`slim` is a Python-pip-only base image with no `conda`. Either pick a
profile that supports the feature (`biomech` for OpenSim, `full` for
PyChrono), or run conda directly on your host outside Docker.

### "I picked 'Don't ask again' and now I can't bring it back"

Edit `~/.upstreamdrift/prefs.json` and remove the
`dont_ask_again.<feature_name>` entry, then restart the launcher.

## Where this all lives

- Feature definitions: [`src/shared/python/feature_registry/features.py`](../../src/shared/python/feature_registry/features.py)
- Probes: [`src/shared/python/engine_core/engine_probes.py`](../../src/shared/python/engine_core/engine_probes.py) + [`src/shared/python/feature_registry/probes.py`](../../src/shared/python/feature_registry/probes.py)
- Install runner: [`src/shared/python/feature_registry/installer.py`](../../src/shared/python/feature_registry/installer.py)
- Re-export shim: [`src/core/capability_registry.py`](../../src/core/capability_registry.py)
- Install dialog: [`src/shared/python/ui/dialogs/install_prompt.py`](../../src/shared/python/ui/dialogs/install_prompt.py)
- Profile catalog: [`docker/profiles.yaml`](../../docker/profiles.yaml)
- Epic / design: [`docs/plans/DOCKER_MODULAR_BUILDS_EPIC.md`](../plans/DOCKER_MODULAR_BUILDS_EPIC.md)
- Operations reference: [`docs/operations/capability-registry.md`](../operations/capability-registry.md)
