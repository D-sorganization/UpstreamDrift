# Docker — Quick Reference

This directory contains Docker-related configuration used by the build system.
For the full rationale behind the three-Dockerfile policy, see
[ADR-0021](../docs/adr/0021-container-strategy.md).

## Dockerfile Roles

| Filename                            | Purpose                                                                                                                                                                                 | Used by CI                                                                                                                   | When to edit                                                            |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `Dockerfile` (repo root)            | **Canonical production build.** Multi-stage (builder / runtime / training). FastAPI on port 8001. Python 3.12-slim with pinned digest.                                                  | `docker-smoke.yml`, `docker-size-gates.yml`, `docker-security-scan.yml`                                                      | Production dependency changes, server config, security hardening        |
| `Dockerfile.heavy_test` (repo root) | **Test environment** mirroring the `d-sorg-fleet-4core` custom runner. Bookworm base with X11/Xvfb, SDL2, and all physics/robotics engines (best-effort).                               | `heavy-tests-opt-in.yml` (runner, not this image)                                                                            | When the runner's apt packages or test-suite requirements change        |
| `Dockerfile.modular` (repo root)    | **Experimental / modular-build validation.** Selects features via `--build-arg PROFILE=<name>` or `--build-arg FEATURES=<list>`. Phase 2 validation vehicle — explicitly non-canonical. | `docker-smoke.yml` (profile capability checks), `docker-size-gates.yml` (profile size matrix) — NOT covered by security scan | Local exploration of modular profiles only; see constraints in ADR-0021 |

## Canonical Release Path

The repo-root `Dockerfile` is the **canonical release** image. Production
releases are built and published only from this file — `Dockerfile.modular` and
`Dockerfile.heavy_test` are never promoted to a release artifact.

```bash
# Build the canonical release image
docker build -t upstream-drift:latest .

# Tag and push a versioned release
docker tag upstream-drift:latest upstream-drift:vX.Y.Z
docker push upstream-drift:vX.Y.Z
```

The canonical `Dockerfile` is the only image covered by the CI security scan
(`docker-security-scan.yml`) and the canonical size gate, so the release
artifact is always the scanned, size-gated build. See
[ADR-0021](../docs/adr/0021-container-strategy.md) for the policy.

## Docker Compose Variants

| File                          | Purpose                                               |
| ----------------------------- | ----------------------------------------------------- |
| `docker-compose.yml`          | Standard local development stack                      |
| `docker-compose.gpu.yml`      | GPU-enabled override (mounts NVIDIA device)           |
| `docker-compose.profiles.yml` | Per-profile services driven by `docker/profiles.yaml` |

## Modular Profiles

Feature profiles and their size budgets are defined in `docker/profiles.yaml`.
The resolver lives in `scripts/docker/install_features.py`.

## Pinned Tools Build Inputs

Modular images package the canonical shared Python roots from the exact
`vendor/ud-tools` gitlink. Because `.git` is intentionally excluded from the
Docker context, the build requires the gitlink revision and a deterministic
path-and-content digest as explicit arguments. CI obtains both through the
`fetch-pinned-tools` action. For a local Bash build, prepare them with:

```bash
git submodule update --init --depth 1 -- vendor/ud-tools
TOOLS_GITLINK_SHA="$(git ls-tree HEAD -- vendor/ud-tools | awk '{print $3}')"
test "$(git -C vendor/ud-tools rev-parse HEAD)" = "$TOOLS_GITLINK_SHA"
TOOLS_SOURCE_SHA256="$(python3 scripts/packaging/pinned_tools_provenance.py \
  --root vendor/ud-tools)"
```

For PowerShell:

```powershell
git submodule update --init --depth 1 -- vendor/ud-tools
$toolsGitlinkSha = ((git ls-tree HEAD -- vendor/ud-tools) -split '\s+')[2]
if ((git -C vendor/ud-tools rev-parse HEAD) -ne $toolsGitlinkSha) {
    throw "Pinned Tools checkout does not match the UpstreamDrift gitlink"
}
$toolsSourceSha256 = python3 scripts/packaging/pinned_tools_provenance.py `
    --root vendor/ud-tools
```

The image build recomputes the digest after copying the required roots and
fails closed if the source bytes, paths, or declared provenance disagree.

## Quick-Start

```bash
# Production image (canonical)
docker build -t upstream-drift:latest .

# Heavy-test image (local runner parity)
docker build -f Dockerfile.heavy_test -t upstream-drift:heavy-test .

# Modular image with a named profile (experimental; Bash variables from above)
docker build --build-arg PROFILE=research \
  --build-arg TOOLS_GITLINK_SHA="$TOOLS_GITLINK_SHA" \
  --build-arg TOOLS_SOURCE_SHA256="$TOOLS_SOURCE_SHA256" \
  -f Dockerfile.modular -t upstream-drift:research .

# Modular image with an explicit feature list (experimental)
docker build --build-arg FEATURES=mujoco,drake,pose-mediapipe \
  --build-arg TOOLS_GITLINK_SHA="$TOOLS_GITLINK_SHA" \
  --build-arg TOOLS_SOURCE_SHA256="$TOOLS_SOURCE_SHA256" \
  -f Dockerfile.modular -t upstream-drift:custom .
```

> **Note:** Only `Dockerfile` is covered by the CI security scan and the canonical
> size gate. `Dockerfile.modular` is covered by the profile smoke tests and the
> per-profile size budget matrix, but NOT the security scan. Changes to it may
> silently diverge from production configuration.
> See [ADR-0021](../docs/adr/0021-container-strategy.md) before promoting
> `Dockerfile.modular` to production use.
