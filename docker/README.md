# Docker — Quick Reference

This directory contains Docker-related configuration used by the build system.
For the full rationale behind the three-Dockerfile policy, see
[ADR-0021](../docs/adr/0021-container-strategy.md).

## Dockerfile Roles

| Filename | Purpose | Used by CI | When to edit |
|---|---|---|---|
| `Dockerfile` (repo root) | **Canonical production build.** Multi-stage (builder / runtime / training). FastAPI on port 8001. Python 3.12-slim with pinned digest. | `docker-smoke.yml`, `docker-size-gates.yml`, `docker-security-scan.yml` | Production dependency changes, server config, security hardening |
| `Dockerfile.heavy_test` (repo root) | **Test environment** mirroring the `d-sorg-fleet-4core` custom runner. Bookworm base with X11/Xvfb, SDL2, and all physics/robotics engines (best-effort). | `heavy-tests-opt-in.yml` (runner, not this image) | When the runner's apt packages or test-suite requirements change |
| `Dockerfile.modular` (repo root) | **Experimental / modular-build validation.** Selects features via `--build-arg PROFILE=<name>` or `--build-arg FEATURES=<list>`. Phase 2 validation vehicle — explicitly non-canonical. | None | Local exploration of modular profiles only; see constraints in ADR-0021 |

## Docker Compose Variants

| File | Purpose |
|---|---|
| `docker-compose.yml` | Standard local development stack |
| `docker-compose.gpu.yml` | GPU-enabled override (mounts NVIDIA device) |
| `docker-compose.profiles.yml` | Per-profile services driven by `docker/profiles.yaml` |

## Modular Profiles

Feature profiles and their size budgets are defined in `docker/profiles.yaml`.
The resolver lives in `scripts/docker/install_features.py`.

## Quick-start

```bash
# Production image (canonical)
docker build -t upstream-drift:latest .

# Heavy-test image (local runner parity)
docker build -f Dockerfile.heavy_test -t upstream-drift:heavy-test .

# Modular image with a named profile (experimental)
docker build --build-arg PROFILE=research -f Dockerfile.modular -t upstream-drift:research .

# Modular image with an explicit feature list (experimental)
docker build --build-arg FEATURES=mujoco,drake,pose-mediapipe -f Dockerfile.modular -t upstream-drift:custom .
```

> **Note:** Only `Dockerfile` is monitored by CI security scans and size gates.
> Changes to `Dockerfile.modular` may silently diverge from production.
> See [ADR-0021](../docs/adr/0021-container-strategy.md) before promoting
> `Dockerfile.modular` to production use.
