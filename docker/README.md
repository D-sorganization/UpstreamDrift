# Root Container Policy

This directory owns the **operational policy** for the three Dockerfiles at the
repo root. The architecture decision is recorded in
[`docs/adr/0021-container-strategy.md`](../docs/adr/0021-container-strategy.md);
this README is the fast "which file do I edit?" guide for day-to-day work.

## Role Matrix

| File | Canonical role | Edit this when... | Do not use it for... |
| --- | --- | --- | --- |
| `Dockerfile` | Default release/runtime/training image | changing the default backend or training image, the base runtime dependency set, or the image used by `docker-compose.yml` / `docker-compose.gpu.yml` | modular profile experiments or heavy-test parity |
| `Dockerfile.heavy_test` | Heavy-test parity image | changing the dependency set mirrored by `scripts/ci/run_local_heavy_tests.sh` or the heavy-test reusable workflow | release/runtime images |
| `Dockerfile.modular` | Opt-in modular profile build surface | changing `docker/profiles.yaml`, feature-registry-driven image composition, or the profile smoke/size CI lanes | the default release/runtime path |

## Compose Mapping

- `docker-compose.yml` uses `Dockerfile` for the default `backend` and
  `training` services.
- `docker-compose.gpu.yml` is an override on top of the default
  `docker-compose.yml` path, so it also inherits `Dockerfile`.
- `docker-compose.profiles.yml` is the supported entry point for
  `Dockerfile.modular` profile images such as `slim`, `standard`, `research`,
  `biomech`, `full`, and `gpu-training`.

## CI Mapping

- `.github/workflows/docker-security-scan.yml` scans the canonical default
  runtime image built from `Dockerfile`.
- `.github/workflows/docker-size-gates.yml` validates both the default
  `Dockerfile` image and the `Dockerfile.modular` profile matrix.
- `.github/workflows/docker-smoke.yml` is dedicated to `Dockerfile.modular`
  capability checks.
- `scripts/check_heavy_dep_parity.py` and
  `scripts/ci/run_local_heavy_tests.sh` guard the `Dockerfile.heavy_test`
  parity contract.

## Edit Rules

1. If you change a root Dockerfile header or workflow role, update this README
   and ADR-0021 in the same PR.
2. If you want `Dockerfile.modular` to replace `Dockerfile` as the default
   runtime image, open a new ADR rather than silently repointing compose or CI.
3. There is currently **no root `Makefile`** to keep in sync for container
   shortcuts; if one is added later, it must follow the same role split.
